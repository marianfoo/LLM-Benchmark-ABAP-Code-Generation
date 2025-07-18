CLASS ltcl_test DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    TYPES: BEGIN OF ty_result,
             even_count TYPE i,
             odd_count  TYPE i,
           END OF ty_result.

    METHODS: test_case_123 FOR TESTING,
      test_case_12  FOR TESTING,
      test_case_3   FOR TESTING,
      test_case_63  FOR TESTING,
      test_case_25  FOR TESTING,
      test_case_19  FOR TESTING,
      test_case_9   FOR TESTING,
      test_case_1   FOR TESTING.
ENDCLASS.

CLASS ltcl_test IMPLEMENTATION.

  METHOD test_case_123.
    DATA result TYPE ty_result.
    result = z_humaneval_107=>even_odd_palindrome( 123 ).
    cl_abap_unit_assert=>assert_equals( act = result-even_count exp = 8 ).
    cl_abap_unit_assert=>assert_equals( act = result-odd_count  exp = 13 ).
  ENDMETHOD.

  METHOD test_case_12.
    DATA result TYPE ty_result.
    result = z_humaneval_107=>even_odd_palindrome( 12 ).
    cl_abap_unit_assert=>assert_equals( act = result-even_count exp = 4 ).
    cl_abap_unit_assert=>assert_equals( act = result-odd_count  exp = 6 ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA result TYPE ty_result.
    result = z_humaneval_107=>even_odd_palindrome( 3 ).
    cl_abap_unit_assert=>assert_equals( act = result-even_count exp = 1 ).
    cl_abap_unit_assert=>assert_equals( act = result-odd_count  exp = 2 ).
  ENDMETHOD.

  METHOD test_case_63.
    DATA result TYPE ty_result.
    result = z_humaneval_107=>even_odd_palindrome( 63 ).
    cl_abap_unit_assert=>assert_equals( act = result-even_count exp = 6 ).
    cl_abap_unit_assert=>assert_equals( act = result-odd_count  exp = 8 ).
  ENDMETHOD.

  METHOD test_case_25.
    DATA result TYPE ty_result.
    result = z_humaneval_107=>even_odd_palindrome( 25 ).
    cl_abap_unit_assert=>assert_equals( act = result-even_count exp = 5 ).
    cl_abap_unit_assert=>assert_equals( act = result-odd_count  exp = 6 ).
  ENDMETHOD.

  METHOD test_case_19.
    DATA result TYPE ty_result.
    result = z_humaneval_107=>even_odd_palindrome( 19 ).
    cl_abap_unit_assert=>assert_equals( act = result-even_count exp = 4 ).
    cl_abap_unit_assert=>assert_equals( act = result-odd_count  exp = 6 ).
  ENDMETHOD.

  METHOD test_case_9.
    DATA result TYPE ty_result.
    result = z_humaneval_107=>even_odd_palindrome( 9 ).
    cl_abap_unit_assert=>assert_equals( act = result-even_count exp = 4 ).
    cl_abap_unit_assert=>assert_equals( act = result-odd_count  exp = 5 ).
  ENDMETHOD.

  METHOD test_case_1.
    DATA result TYPE ty_result.
    result = z_humaneval_107=>even_odd_palindrome( 1 ).
    cl_abap_unit_assert=>assert_equals( act = result-even_count exp = 0 ).
    cl_abap_unit_assert=>assert_equals( act = result-odd_count  exp = 1 ).
  ENDMETHOD.

ENDCLASS.