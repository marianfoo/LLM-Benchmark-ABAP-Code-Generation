CLASS z_humaneval_162 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: string_to_md5
      IMPORTING
        text          TYPE string
      RETURNING
        VALUE(result) TYPE string.

ENDCLASS.

CLASS z_humaneval_162 IMPLEMENTATION.

  METHOD string_to_md5.

    DATA: lv_hash TYPE hash160.

    IF text IS INITIAL.
      CLEAR result.
      RETURN.
    ENDIF.

    CALL FUNCTION 'CALCULATE_HASH_FOR_CHAR'
      EXPORTING
        alg  = 'MD5'
        data = text
      IMPORTING
        hash = lv_hash.

    result = lv_hash.
    TRANSLATE result TO LOWER CASE.

  ENDMETHOD.

ENDCLASS.