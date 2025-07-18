CLASS z_humaneval_007_test DEFINITION
FOR TESTING
INHERITING FROM cl_aunit_assert
FINAL
RISK LEVEL HARMLESS
CREATE PUBLIC.

  PUBLIC SECTION.
    METHODS: test_empty_input FOR TESTING,
      test_filter_xxx FOR TESTING,
      test_filter_xx FOR TESTING,
      test_filter_run FOR TESTING.

  PRIVATE SECTION.
    METHODS setup.
ENDCLASS.

CLASS z_humaneval_007_test IMPLEMENTATION.

  METHOD setup.
  ENDMETHOD.

  METHOD test_empty_input.
    DATA: result          TYPE string_table,
          input_strings   TYPE string_table,
          expected_result TYPE string_table.

    CLEAR: input_strings, expected_result.
    result = z_humaneval_007=>filter_by_substring(
    strings = input_strings
    substring = 'john' ).

    assert_equals( act = result exp = expected_result ).
  ENDMETHOD.

  METHOD test_filter_xxx.
    DATA: result          TYPE string_table,
          input_strings   TYPE string_table,
          expected_result TYPE string_table.

    CLEAR: input_strings, expected_result.
    APPEND 'xxx' TO input_strings.
    APPEND 'asd' TO input_strings.
    APPEND 'xxy' TO input_strings.
    APPEND 'john doe' TO input_strings.
    APPEND 'xxxAAA' TO input_strings.
    APPEND 'xxx' TO input_strings.

    APPEND 'xxx' TO expected_result.
    APPEND 'xxxAAA' TO expected_result.
    APPEND 'xxx' TO expected_result.

    result = z_humaneval_007=>filter_by_substring(
    strings = input_strings
    substring = 'xxx' ).

    assert_equals( act = result exp = expected_result ).
  ENDMETHOD.

  METHOD test_filter_xx.
    DATA: result          TYPE string_table,
          input_strings   TYPE string_table,
          expected_result TYPE string_table.

    CLEAR: input_strings, expected_result.
    APPEND 'xxx' TO input_strings.
    APPEND 'asd' TO input_strings.
    APPEND 'aaaxxy' TO input_strings.
    APPEND 'john doe' TO input_strings.
    APPEND 'xxxAAA' TO input_strings.
    APPEND 'xxx' TO input_strings.

    APPEND 'xxx' TO expected_result.
    APPEND 'aaaxxy' TO expected_result.
    APPEND 'xxxAAA' TO expected_result.
    APPEND 'xxx' TO expected_result.

    result = z_humaneval_007=>filter_by_substring(
    strings = input_strings
    substring = 'xx' ).

    assert_equals( act = result exp = expected_result ).
  ENDMETHOD.

  METHOD test_filter_run.
    DATA: result          TYPE string_table,
          input_strings   TYPE string_table,
          expected_result TYPE string_table.

    CLEAR: input_strings, expected_result.
    APPEND 'grunt' TO input_strings.
    APPEND 'trumpet' TO input_strings.
    APPEND 'prune' TO input_strings.
    APPEND 'gruesome' TO input_strings.

    APPEND 'grunt' TO expected_result.
    APPEND 'prune' TO expected_result.

    result = z_humaneval_007=>filter_by_substring(
    strings = input_strings
    substring = 'run' ).

    assert_equals( act = result exp = expected_result ).
  ENDMETHOD.

ENDCLASS.