CLASS lcl_test_iscube DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

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

CLASS lcl_test_iscube IMPLEMENTATION.

  METHOD test_case_1.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_077=>iscube( 1 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_true msg = 'First test error' ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_077=>iscube( 2 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_false msg = 'Second test error' ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_077=>iscube( -1 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_true msg = 'Third test error' ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_077=>iscube( 64 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_true msg = 'Fourth test error' ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_077=>iscube( 180 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_false msg = 'Fifth test error' ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_077=>iscube( 1000 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_true msg = 'Sixth test error' ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_077=>iscube( 0 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_true msg = '1st edge test error' ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_077=>iscube( 1729 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_false msg = '2nd edge test error' ).
  ENDMETHOD.

ENDCLASS.