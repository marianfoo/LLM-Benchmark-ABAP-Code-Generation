CLASS z_humaneval_006_test DEFINITION
  FOR TESTING
  FINAL
  CREATE PUBLIC
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING.
ENDCLASS.

CLASS z_humaneval_006_test IMPLEMENTATION.

  METHOD test_case_1.
    DATA: lt_result   TYPE int4_table,
          lt_expected TYPE int4_table.

    lt_result = z_humaneval_006=>parse_nested_parens( paren_string = '(()()) ((())) () ((())()())' ).

    CLEAR lt_expected.
    APPEND: 2 TO lt_expected,
            3 TO lt_expected,
            1 TO lt_expected,
            3 TO lt_expected.

    cl_aunit_assert=>assert_equals(
      act = lt_result
      exp = lt_expected
      msg = 'Test case 1 failed'
    ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: lt_result   TYPE int4_table,
          lt_expected TYPE int4_table.

    lt_result = z_humaneval_006=>parse_nested_parens( paren_string = '() (()) ((())) (((())))' ).

    CLEAR lt_expected.
    APPEND: 1 TO lt_expected,
            2 TO lt_expected,
            3 TO lt_expected,
            4 TO lt_expected.

    cl_aunit_assert=>assert_equals(
      act = lt_result
      exp = lt_expected
      msg = 'Test case 2 failed'
    ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: lt_result   TYPE int4_table,
          lt_expected TYPE int4_table.

    lt_result = z_humaneval_006=>parse_nested_parens( paren_string = '(()(())((())))' ).

    CLEAR lt_expected.
    APPEND 4 TO lt_expected.

    cl_aunit_assert=>assert_equals(
      act = lt_result
      exp = lt_expected
      msg = 'Test case 3 failed'
    ).
  ENDMETHOD.

ENDCLASS.