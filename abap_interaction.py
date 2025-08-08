from html import unescape
import json
from typing import List, Literal
from abap_adt_py.adt_client import AdtClient
import xml.etree.ElementTree as ET
import re
import tqdm

MODELS_TO_RUN = [
    "llama-3.3-70b-instruct",
    "qwen2.5-coder-32b-instruct",
    "codestral-22b",
]


def get_unittest_src(unittest_file: str):
    with open(unittest_file, "r") as file:
        test_src = file.read()
        return test_src


def unittest_find_method_calls(
    default_class_name: str, unittest_file: str
) -> List[str]:
    test_src = get_unittest_src(unittest_file)
    finds = list(set(re.findall(f"{default_class_name}=>.*\\(", test_src)))
    return finds


def class_find_method_calls(client: AdtClient, class_name: str):
    class_uri = f"/sap/bc/adt/oo/classes/{class_name}"
    resp = client.object_structure(class_uri)
    methods = [
        m
        for m in resp["components"]
        if "type" in m
        and m["type"] == "CLAS/OM"
        and "visibility" in m
        and m["visibility"] == "public"
    ]
    method_names = [m["name"] for m in methods if "name" in m]
    return method_names


def build_unittest_src(class_name: str, unittest_file: str):
    default_class_name = f"z_humaneval_{str.zfill(prompt_file[:-4], 3)}"
    unittest_file = f"abap_unittests/{default_class_name}_test.abap"

    unittest_method_calls = unittest_find_method_calls(
        default_class_name, unittest_file
    )
    class_methods = class_find_method_calls(client, class_name)

    if len(unittest_method_calls) == 1 and len(class_methods) == 1:
        unittest_src = get_unittest_src(unittest_file)
        new_method_call = f"{class_name}=>{class_methods[0]}("
        unittest_src = unittest_src.replace(unittest_method_calls[0], new_method_call)
        # print(f"replaced {unittest_method_calls[0]} with {new_method_call}")
        return unittest_src
    else:
        add_to_chat(chat, "There should only be the one public method.")
        return ""


def run_unit_tests(client: AdtClient, class_uri: str):
    result = client.run_unit_test(class_uri)

    if len(result) > 0:
        add_to_chat(
            chat,
            f"The unit test failed with the following errors:\n{result}\n Try to fix the class, not the unit test.",
        )
        success = False
    else:
        add_to_chat(
            chat,
            f"The unit tests were successful.",
        )
        success = True
    return success


def create_test_class_include(client: AdtClient, class_name: str):
    try:
        lock_handle: str = client.lock(class_uri)
        client.create_test_class_include(class_name, lock_handle)
    finally:
        client.unlock(class_uri, lock_handle)


def syntax_check(client: AdtClient, class_uri: str, src: str) -> bool:
    syntaxcheck_results = client.syntax_check(class_uri, class_uri, src)
    if len(syntaxcheck_results) > 0:
        add_to_chat(
            chat,
            f"The syntax check failed with the following errors:\n{syntaxcheck_results}",
        )
        success = False
    else:
        success = True
    return success


def syntax_check_unittest(client: AdtClient, class_uri: str, src: str) -> bool:
    syntaxcheck_results = client.syntax_check(class_uri, class_uri, src)
    if len(syntaxcheck_results) > 0:
        add_to_chat(
            chat,
            f"The unittest syntax check failed with the following errors:\n{syntaxcheck_results}\nTry to fix the class, not the unit test.",
        )
        success = False
    else:
        success = True
    return success


def create_class(client: AdtClient, class_name: str) -> bool:
    try:
        client.create(
            object_type="CLAS/OC",
            name=class_name,
            description="Automatically created class for benchmarking purposes",
            parent="$TMP",
        )
        success = True
    except Exception as e:
        error_text = str(e)
        error_text = error_text[error_text.index("\n") + 1 :].strip()
        xml_error = parse_error_xml(error_text)
        add_to_chat(
            chat,
            f"The class could not be created due to the following error:\n{xml_error}",
        )
        success = False
    return success


def set_source(
    client: AdtClient,
    class_uri: str,
    src: str,
    suffix: Literal["/source/main", "/includes/testclasses"] = "/source/main",
):
    try:
        lock_handle: str = client.lock(class_uri)
        try:
            client.set_object_source(f"{class_uri}{suffix}", src, lock_handle)
            success = True
        finally:
            client.unlock(class_uri, lock_handle)
    except Exception as e:
        error_text = str(e)
        error_text = error_text[error_text.index("\n") + 1 :]
        xml_error = parse_error_xml(error_text)
        add_to_chat(
            chat,
            f"The source code could not be set due to the following error:\n{xml_error}",
        )
        success = False
    return success


def activate_class(client: AdtClient, class_uri: str) -> bool:
    try:
        client.activate(class_uri, class_uri)
        success = True
    except Exception as e:
        error_text = str(e)
        add_to_chat(
            chat,
            f"The activation failed with the following error:\n{error_text}",
        )
        success = False
    return success


def delete_class(client: AdtClient, class_uri: str):
    lock_handle: str = client.lock(class_uri)
    try:
        client.delete(class_uri, lock_handle)
    finally:
        client.unlock(class_uri, lock_handle)


def clean_longtext(longtext_html: str) -> str:
    clean_text = ""
    if longtext_html:
        decoded_html = unescape(longtext_html)
        try:
            text = re.sub(r"<br\s*/?>", "\n", decoded_html, flags=re.IGNORECASE)
            text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
            text = re.sub(r"<p[^>]*>", "", text, flags=re.IGNORECASE)
            text = re.sub(r"<[^>]+>", "", text)
            lines = [line.strip() for line in text.split("\n")]
            cleaned_lines = []
            prev_empty = False
            for line in lines:
                if line:
                    cleaned_lines.append(line)
                    prev_empty = False
                elif not prev_empty:
                    cleaned_lines.append("")
                    prev_empty = True
            clean_text = "\n".join(cleaned_lines).strip()
        except:
            clean_text = decoded_html
    return clean_text


def parse_error_xml(xml_string):
    root = ET.fromstring(xml_string)

    exception_type = (
        root.find(".//type").get("id") if root.find(".//type") is not None else None,  # type: ignore
    )[0]
    message = (
        root.find(".//message").text if root.find(".//message") is not None else None,  # type: ignore
    )[0]

    properties = {}
    for entry in root.findall(".//entry"):
        key = entry.get("key")
        value = entry.text
        properties[key] = value

    longtext_html = properties.get("LONGTEXT", "")
    clean_text = clean_longtext(longtext_html)

    return {
        "exception_type": exception_type,
        "message": message,
        "longtext": clean_text,
    }


def add_to_chat(chat, content):
    chat.append(
        {
            "role": "user",
            "content": content,
        }
    )


if __name__ == "__main__":
    client: AdtClient = AdtClient(
        sap_host="http://localhost:50000",
        username="DEVELOPER",
        password="ABAPtr2022#01",
        client="001",
        language="EN",
    )
    client.login()

    for model_name in tqdm.tqdm(MODELS_TO_RUN, desc=f"Processing LLMs"):
        filename = f"{model_name}.json"
        with open(filename, "r") as file:
            prompt_files = json.load(file)
        for prompt_file, chats in tqdm.tqdm(
            prompt_files.items(), "Processing prompts", leave=False
        ):

            for chat in tqdm.tqdm(chats, desc=f"Processing {prompt_file}", leave=False):
                skip_remaining_steps = False
                last_message = chat[-1]
                src = last_message["content"]
                class_name = src.split(" ")[1].strip(".")
                class_uri = f"/sap/bc/adt/oo/classes/{class_name}"

                success = create_class(client, class_name)
                if not success:
                    skip_remaining_steps = True

                if not skip_remaining_steps:
                    success = syntax_check(client, class_uri, src)
                    if not success:
                        skip_remaining_steps = True

                if not skip_remaining_steps:
                    success = set_source(client, class_uri, src)
                    if not success:
                        skip_remaining_steps = True

                if not skip_remaining_steps:
                    success = activate_class(client, class_uri)
                    if not success:
                        skip_remaining_steps = True

                if not skip_remaining_steps:
                    create_test_class_include(client, class_name)
                    unittest_src = build_unittest_src(class_name, prompt_file)
                    if unittest_src == "":
                        skip_remaining_steps = True

                    success = syntax_check_unittest(
                        client, class_uri + "/includes/testclasses", unittest_src
                    )
                    if not success:
                        skip_remaining_steps = True

                    if not skip_remaining_steps:
                        success = set_source(
                            client, class_uri, unittest_src, "/includes/testclasses"
                        )
                        success = True
                        if not success:
                            skip_remaining_steps = True

                    if not skip_remaining_steps:
                        success = activate_class(client, class_uri)
                        if not success:
                            skip_remaining_steps = True

                    if not skip_remaining_steps:
                        success = run_unit_tests(client, class_uri)

                try:
                    delete_class(client, class_uri)
                except Exception as e:
                    pass

                with open(filename, "w") as file:
                    file.write(json.dumps(prompt_files, indent=4, ensure_ascii=False))
