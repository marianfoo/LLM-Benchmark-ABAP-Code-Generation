CLASS ltcl_test_largest_prime_factor DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS test_15 FOR TESTING.
    METHODS test_27 FOR TESTING.
    METHODS test_63 FOR TESTING.
    METHODS test_330 FOR TESTING.
    METHODS test_13195 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_largest_prime_factor IMPLEMENTATION.

  METHOD test_15.
    DATA: result TYPE i.
    result = z_humaneval_059=>largest_prime_factor( 15 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 5 ).
  ENDMETHOD.

  METHOD test_27.
    DATA: result TYPE i.
    result = z_humaneval_059=>largest_prime_factor( 27 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 3 ).
  ENDMETHOD.

  METHOD test_63.
    DATA: result TYPE i.
    result = z_humaneval_059=>largest_prime_factor( 63 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 7 ).
  ENDMETHOD.

  METHOD test_330.
    DATA: result TYPE i.
    result = z_humaneval_059=>largest_prime_factor( 330 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 11 ).
  ENDMETHOD.

  METHOD test_13195.
    DATA: result TYPE i.
    result = z_humaneval_059=>largest_prime_factor( 13195 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 29 ).
  ENDMETHOD.

ENDCLASS.