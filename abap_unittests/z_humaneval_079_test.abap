CLASS ltcl_test_decimal_to_binary DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_0 FOR TESTING,
      test_case_32 FOR TESTING,
      test_case_103 FOR TESTING,
      test_case_15 FOR TESTING,
      test_edge_cases FOR TESTING.

ENDCLASS.

CLASS ltcl_test_decimal_to_binary IMPLEMENTATION.

  METHOD test_case_0.
    DATA: lv_result TYPE string.

    lv_result = z_humaneval_079=>decimal_to_binary( 0 ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 'db0db'
    ).
  ENDMETHOD.

  METHOD test_case_32.
    DATA: lv_result TYPE string.

    lv_result = z_humaneval_079=>decimal_to_binary( 32 ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 'db100000db'
    ).
  ENDMETHOD.

  METHOD test_case_103.
    DATA: lv_result TYPE string.

    lv_result = z_humaneval_079=>decimal_to_binary( 103 ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 'db1100111db'
    ).
  ENDMETHOD.

  METHOD test_case_15.
    DATA: lv_result TYPE string.

    lv_result = z_humaneval_079=>decimal_to_binary( 15 ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 'db1111db'
      msg = 'This prints if this assert fails 1 (good for debugging!)'
    ).
  ENDMETHOD.

  METHOD test_edge_cases.
    cl_abap_unit_assert=>assert_true(
      act = abap_true
      msg = 'This prints if this assert fails 2 (also good for debugging!)'
    ).
  ENDMETHOD.

ENDCLASS.