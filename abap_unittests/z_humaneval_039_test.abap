CLASS ltcl_test_prime_fib DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING,
      test_case_8 FOR TESTING,
      test_case_9 FOR TESTING,
      test_case_10 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_prime_fib IMPLEMENTATION.
  METHOD test_case_1.
    DATA: result TYPE i.
    result = z_humaneval_039=>prime_fib( 1 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 2 ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: result TYPE i.
    result = z_humaneval_039=>prime_fib( 2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 3 ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: result TYPE i.
    result = z_humaneval_039=>prime_fib( 3 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 5 ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: result TYPE i.
    result = z_humaneval_039=>prime_fib( 4 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 13 ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: result TYPE i.
    result = z_humaneval_039=>prime_fib( 5 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 89 ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: result TYPE i.
    result = z_humaneval_039=>prime_fib( 6 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 233 ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: result TYPE i.
    result = z_humaneval_039=>prime_fib( 7 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 1597 ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: result TYPE i.
    result = z_humaneval_039=>prime_fib( 8 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 28657 ).
  ENDMETHOD.

  METHOD test_case_9.
    DATA: result TYPE i.
    result = z_humaneval_039=>prime_fib( 9 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 514229 ).
  ENDMETHOD.

  METHOD test_case_10.
    DATA: result TYPE i.
    result = z_humaneval_039=>prime_fib( 10 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 433494437 ).
  ENDMETHOD.
ENDCLASS.