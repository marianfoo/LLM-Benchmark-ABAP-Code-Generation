CLASS ltcl_test_z_humaneval_105 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS:
      test_basic_case FOR TESTING,
      test_empty_array FOR TESTING,
      test_with_invalid_numbers FOR TESTING,
      test_mixed_valid_invalid FOR TESTING,
      test_descending_order FOR TESTING.

ENDCLASS.

CLASS ltcl_test_z_humaneval_105 IMPLEMENTATION.

  METHOD test_basic_case.
    DATA: lt_input    TYPE TABLE OF i,
          lt_expected TYPE string_table,
          lt_actual   TYPE string_table.

    APPEND: 2 TO lt_input, 1 TO lt_input, 1 TO lt_input, 4 TO lt_input,
            5 TO lt_input, 8 TO lt_input, 2 TO lt_input, 3 TO lt_input.

    APPEND: 'Eight' TO lt_expected, 'Five' TO lt_expected, 'Four' TO lt_expected,
            'Three' TO lt_expected, 'Two' TO lt_expected, 'Two' TO lt_expected,
            'One' TO lt_expected, 'One' TO lt_expected.

    lt_actual = z_humaneval_105=>by_length( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lt_actual
      exp = lt_expected
      msg = 'Basic case failed' ).
  ENDMETHOD.

  METHOD test_empty_array.
    DATA: lt_input    TYPE TABLE OF i,
          lt_expected TYPE string_table,
          lt_actual   TYPE string_table.

    lt_actual = z_humaneval_105=>by_length( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lt_actual
      exp = lt_expected
      msg = 'Empty array case failed' ).
  ENDMETHOD.

  METHOD test_with_invalid_numbers.
    DATA: lt_input    TYPE TABLE OF i,
          lt_expected TYPE string_table,
          lt_actual   TYPE string_table.

    APPEND: 1 TO lt_input, -1 TO lt_input, 55 TO lt_input.

    APPEND: 'One' TO lt_expected.

    lt_actual = z_humaneval_105=>by_length( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lt_actual
      exp = lt_expected
      msg = 'Invalid numbers case failed' ).
  ENDMETHOD.

  METHOD test_mixed_valid_invalid.
    DATA: lt_input    TYPE TABLE OF i,
          lt_expected TYPE string_table,
          lt_actual   TYPE string_table.

    APPEND: 1 TO lt_input, -1 TO lt_input, 3 TO lt_input, 2 TO lt_input.

    APPEND: 'Three' TO lt_expected, 'Two' TO lt_expected, 'One' TO lt_expected.

    lt_actual = z_humaneval_105=>by_length( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lt_actual
      exp = lt_expected
      msg = 'Mixed valid invalid case failed' ).
  ENDMETHOD.

  METHOD test_descending_order.
    DATA: lt_input    TYPE TABLE OF i,
          lt_expected TYPE string_table,
          lt_actual   TYPE string_table.

    APPEND: 9 TO lt_input, 4 TO lt_input, 8 TO lt_input.

    APPEND: 'Nine' TO lt_expected, 'Eight' TO lt_expected, 'Four' TO lt_expected.

    lt_actual = z_humaneval_105=>by_length( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lt_actual
      exp = lt_expected
      msg = 'Descending order case failed' ).
  ENDMETHOD.

ENDCLASS.