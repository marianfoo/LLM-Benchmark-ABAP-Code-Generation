CLASS ltcl_test_compare_one DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING,
      test_case_8 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_compare_one IMPLEMENTATION.

  METHOD test_case_1.
    DATA: result TYPE string.
    result = z_humaneval_137=>compare_one( a = 1 b = 2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '2' ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: result TYPE string.
    result = z_humaneval_137=>compare_one( a = 1 b = '2.5' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '2.5' ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: result TYPE string.
    result = z_humaneval_137=>compare_one( a = 2 b = 3 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '3' ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: result TYPE string.
    result = z_humaneval_137=>compare_one( a = 5 b = 6 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '6' ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: result TYPE string.
    result = z_humaneval_137=>compare_one( a = 1 b = '2,3' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '2,3' ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: result TYPE string.
    result = z_humaneval_137=>compare_one( a = '5,1' b = '6' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '6' ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: result TYPE string.
    result = z_humaneval_137=>compare_one( a = '1' b = '2' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '2' ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: result TYPE string.
    result = z_humaneval_137=>compare_one( a = '1' b = 1 ).
    cl_abap_unit_assert=>assert_initial( result ).
  ENDMETHOD.

ENDCLASS.