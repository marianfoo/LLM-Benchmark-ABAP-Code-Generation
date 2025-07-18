CLASS ltcl_test_digits DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_5 FOR TESTING,
      test_54 FOR TESTING,
      test_120 FOR TESTING,
      test_5014 FOR TESTING,
      test_98765 FOR TESTING,
      test_5576543 FOR TESTING,
      test_2468 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_digits IMPLEMENTATION.

  METHOD test_5.
    DATA: result TYPE i.
    result = z_humaneval_131=>digits( 5 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 5 ).
  ENDMETHOD.

  METHOD test_54.
    DATA: result TYPE i.
    result = z_humaneval_131=>digits( 54 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 5 ).
  ENDMETHOD.

  METHOD test_120.
    DATA: result TYPE i.
    result = z_humaneval_131=>digits( 120 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 1 ).
  ENDMETHOD.

  METHOD test_5014.
    DATA: result TYPE i.
    result = z_humaneval_131=>digits( 5014 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 5 ).
  ENDMETHOD.

  METHOD test_98765.
    DATA: result TYPE i.
    result = z_humaneval_131=>digits( 98765 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 315 ).
  ENDMETHOD.

  METHOD test_5576543.
    DATA: result TYPE i.
    result = z_humaneval_131=>digits( 5576543 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 2625 ).
  ENDMETHOD.

  METHOD test_2468.
    DATA: result TYPE i.
    result = z_humaneval_131=>digits( 2468 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 0 ).
  ENDMETHOD.

ENDCLASS.