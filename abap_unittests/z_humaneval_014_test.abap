CLASS z_humaneval_014_DEMO DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS
  FINAL
  CREATE PUBLIC.

  PRIVATE SECTION.
    METHODS:
      test_empty_string FOR TESTING,
      test_normal_string FOR TESTING,
      test_repeated_characters FOR TESTING.
ENDCLASS.

CLASS z_humaneval_014_DEMO IMPLEMENTATION.

  METHOD test_empty_string.
    DATA: lt_result TYPE string_table.


    lt_result = z_humaneval_014=>all_prefixes( string = '' ).

    cl_abap_unit_assert=>assert_equals(
      act = lines( lt_result )
      exp = 0
      msg = 'Empty string should return empty table' ).
  ENDMETHOD.

  METHOD test_normal_string.
    DATA: lt_result   TYPE string_table,
          lt_expected TYPE string_table.

    lt_result = z_humaneval_014=>all_prefixes( string = 'asdfgh' ).

    APPEND 'a' TO lt_expected.
    APPEND 'as' TO lt_expected.
    APPEND 'asd' TO lt_expected.
    APPEND 'asdf' TO lt_expected.
    APPEND 'asdfg' TO lt_expected.
    APPEND 'asdfgh' TO lt_expected.

    cl_abap_unit_assert=>assert_equals(
      act = lt_result
      exp = lt_expected
      msg = 'Normal string prefixes test failed' ).
  ENDMETHOD.

  METHOD test_repeated_characters.
    DATA: lt_result   TYPE string_table,
          lt_expected TYPE string_table.


    lt_result = z_humaneval_014=>all_prefixes( string = 'WWW' ).

    APPEND 'W' TO lt_expected.
    APPEND 'WW' TO lt_expected.
    APPEND 'WWW' TO lt_expected.

    cl_abap_unit_assert=>assert_equals(
      act = lt_result
      exp = lt_expected
      msg = 'Repeated characters test failed' ).
  ENDMETHOD.

ENDCLASS.