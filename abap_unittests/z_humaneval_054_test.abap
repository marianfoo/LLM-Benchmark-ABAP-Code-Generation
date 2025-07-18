CLASS ltcl_test_same_chars DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_same_chars IMPLEMENTATION.

  METHOD test_case_1.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_054=>same_chars( s0 = 'eabcdzzzz' s1 = 'dddzzzzzzzddeddabc' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_054=>same_chars( s0 = 'abcd' s1 = 'dddddddabc' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_054=>same_chars( s0 = 'dddddddabc' s1 = 'abcd' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_054=>same_chars( s0 = 'eabcd' s1 = 'dddddddabc' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_054=>same_chars( s0 = 'abcd' s1 = 'dddddddabcf' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_054=>same_chars( s0 = 'eabcdzzzz' s1 = 'dddzzzzzzzddddabc' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: lv_result TYPE abap_bool.
    lv_result = z_humaneval_054=>same_chars( s0 = 'aabb' s1 = 'aaccc' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = abap_false ).
  ENDMETHOD.

ENDCLASS.