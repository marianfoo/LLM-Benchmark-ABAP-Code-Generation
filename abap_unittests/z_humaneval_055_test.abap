CLASS ltcl_test_fib DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    METHODS: test_fib_10 FOR TESTING,
      test_fib_1 FOR TESTING,
      test_fib_8 FOR TESTING,
      test_fib_11 FOR TESTING,
      test_fib_12 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_fib IMPLEMENTATION.
  METHOD test_fib_10.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_055=>fib( 10 )
      exp = 55 ).
  ENDMETHOD.

  METHOD test_fib_1.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_055=>fib( 1 )
      exp = 1 ).
  ENDMETHOD.

  METHOD test_fib_8.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_055=>fib( 8 )
      exp = 21 ).
  ENDMETHOD.

  METHOD test_fib_11.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_055=>fib( 11 )
      exp = 89 ).
  ENDMETHOD.

  METHOD test_fib_12.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_055=>fib( 12 )
      exp = 144 ).
  ENDMETHOD.
ENDCLASS.