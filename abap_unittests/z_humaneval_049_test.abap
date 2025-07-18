CLASS ltcl_test_modp DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_modp_3_5 FOR TESTING,
      test_modp_1101_101 FOR TESTING,
      test_modp_0_101 FOR TESTING,
      test_modp_3_11 FOR TESTING,
      test_modp_100_101 FOR TESTING,
      test_modp_30_5 FOR TESTING,
      test_modp_31_5 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_modp IMPLEMENTATION.

  METHOD test_modp_3_5.
    DATA: result TYPE i.
    result = z_humaneval_049=>modp( n = 3 p = 5 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 3 ).
  ENDMETHOD.

  METHOD test_modp_1101_101.
    DATA: result TYPE i.
    result = z_humaneval_049=>modp( n = 1101 p = 101 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 2 ).
  ENDMETHOD.

  METHOD test_modp_0_101.
    DATA: result TYPE i.
    result = z_humaneval_049=>modp( n = 0 p = 101 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 1 ).
  ENDMETHOD.

  METHOD test_modp_3_11.
    DATA: result TYPE i.
    result = z_humaneval_049=>modp( n = 3 p = 11 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 8 ).
  ENDMETHOD.

  METHOD test_modp_100_101.
    DATA: result TYPE i.
    result = z_humaneval_049=>modp( n = 100 p = 101 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 1 ).
  ENDMETHOD.

  METHOD test_modp_30_5.
    DATA: result TYPE i.
    result = z_humaneval_049=>modp( n = 30 p = 5 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 4 ).
  ENDMETHOD.

  METHOD test_modp_31_5.
    DATA: result TYPE i.
    result = z_humaneval_049=>modp( n = 31 p = 5 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 3 ).
  ENDMETHOD.

ENDCLASS.