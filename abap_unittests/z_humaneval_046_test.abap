CLASS ltcl_test_z_humaneval_046 DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    METHODS: test_fib4_5 FOR TESTING,
      test_fib4_8 FOR TESTING,
      test_fib4_10 FOR TESTING,
      test_fib4_12 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_z_humaneval_046 IMPLEMENTATION.
  METHOD test_fib4_5.
    DATA: result TYPE i.
    result = z_humaneval_046=>fib4( 5 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 4 ).
  ENDMETHOD.

  METHOD test_fib4_8.
    DATA: result TYPE i.
    result = z_humaneval_046=>fib4( 8 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 28 ).
  ENDMETHOD.

  METHOD test_fib4_10.
    DATA: result TYPE i.
    result = z_humaneval_046=>fib4( 10 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 104 ).
  ENDMETHOD.

  METHOD test_fib4_12.
    DATA: result TYPE i.
    result = z_humaneval_046=>fib4( 12 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 386 ).
  ENDMETHOD.
ENDCLASS.