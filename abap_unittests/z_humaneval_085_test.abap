CLASS Zcl_HUMANEVAL_085 DEFINITION FOR TESTING
  RISK LEVEL HARMLESS.
  PUBLIC SECTION.
    METHODS:
      test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING.
ENDCLASS.

CLASS Zcl_HUMANEVAL_085 IMPLEMENTATION.

  METHOD test_case_1.
    DATA(lv_result) = z_humaneval_085=>add( VALUE #( ( 4 ) ( 88 ) ) ).
    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 88 ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA(lv_result) = z_humaneval_085=>add( VALUE #( ( 4 ) ( 5 ) ( 6 ) ( 7 ) ( 2 ) ( 122 ) ) ).
    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 122 ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA(lv_result) = z_humaneval_085=>add( VALUE #( ( 4 ) ( 0 ) ( 6 ) ( 7 ) ) ).
    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 0 ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA(lv_result) = z_humaneval_085=>add( VALUE #( ( 4 ) ( 4 ) ( 6 ) ( 8 ) ) ).
    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 12 ).
  ENDMETHOD.

ENDCLASS.