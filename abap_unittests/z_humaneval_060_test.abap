CLASS ltcl_test_sum_to_n DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    METHODS: test_sum_to_n_1 FOR TESTING,
      test_sum_to_n_6 FOR TESTING,
      test_sum_to_n_11 FOR TESTING,
      test_sum_to_n_30 FOR TESTING,
      test_sum_to_n_100 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_sum_to_n IMPLEMENTATION.
  METHOD test_sum_to_n_1.
    DATA: lv_result TYPE i.
    lv_result = z_humaneval_060=>sum_to_n( 1 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 1 ).
  ENDMETHOD.

  METHOD test_sum_to_n_6.
    DATA: lv_result TYPE i.
    lv_result = z_humaneval_060=>sum_to_n( 6 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 21 ).
  ENDMETHOD.

  METHOD test_sum_to_n_11.
    DATA: lv_result TYPE i.
    lv_result = z_humaneval_060=>sum_to_n( 11 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 66 ).
  ENDMETHOD.

  METHOD test_sum_to_n_30.
    DATA: lv_result TYPE i.
    lv_result = z_humaneval_060=>sum_to_n( 30 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 465 ).
  ENDMETHOD.

  METHOD test_sum_to_n_100.
    DATA: lv_result TYPE i.
    lv_result = z_humaneval_060=>sum_to_n( 100 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 5050 ).
  ENDMETHOD.
ENDCLASS.