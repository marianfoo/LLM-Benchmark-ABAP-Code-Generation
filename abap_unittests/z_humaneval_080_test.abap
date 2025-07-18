CLASS ltcl_test_is_happy DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_a FOR TESTING,
      test_case_aa FOR TESTING,
      test_case_abcd FOR TESTING,
      test_case_aabb FOR TESTING,
      test_case_adb FOR TESTING,
      test_case_xyy FOR TESTING,
      test_case_iopaxpoi FOR TESTING,
      test_case_iopaxioi FOR TESTING.

ENDCLASS.

CLASS ltcl_test_is_happy IMPLEMENTATION.

  METHOD test_case_a.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_080=>is_happy( 'a' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_aa.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_080=>is_happy( 'aa' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_abcd.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_080=>is_happy( 'abcd' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_aabb.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_080=>is_happy( 'aabb' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_adb.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_080=>is_happy( 'adb' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_xyy.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_080=>is_happy( 'xyy' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_iopaxpoi.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_080=>is_happy( 'iopaxpoi' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_iopaxioi.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_080=>is_happy( 'iopaxioi' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_false ).
  ENDMETHOD.

ENDCLASS.