CLASS ltc_test_sum_squares DEFINITION
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
      test_case_8 FOR TESTING,
      test_case_9 FOR TESTING,
      test_case_10 FOR TESTING,
      test_case_11 FOR TESTING.

ENDCLASS.

CLASS ltc_test_sum_squares IMPLEMENTATION.

  METHOD test_case_1.
    DATA: lt_input  TYPE TABLE OF i,
          lv_result TYPE i.

    lt_input = VALUE #( ( 1 ) ( 2 ) ( 3 ) ).
    lv_result = z_humaneval_142=>sum_squares( lt_input ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 6 ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: lt_input  TYPE TABLE OF i,
          lv_result TYPE i.

    lt_input = VALUE #( ( 1 ) ( 4 ) ( 9 ) ).
    lv_result = z_humaneval_142=>sum_squares( lt_input ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 14 ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: lt_input  TYPE TABLE OF i,
          lv_result TYPE i.

    lt_input = VALUE #( ).
    lv_result = z_humaneval_142=>sum_squares( lt_input ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 0 ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: lt_input  TYPE TABLE OF i,
          lv_result TYPE i.

    lt_input = VALUE #( ( 1 ) ( 1 ) ( 1 ) ( 1 ) ( 1 ) ( 1 ) ( 1 ) ( 1 ) ( 1 ) ).
    lv_result = z_humaneval_142=>sum_squares( lt_input ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 9 ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: lt_input  TYPE TABLE OF i,
          lv_result TYPE i.

    lt_input = VALUE #( ( -1 ) ( -1 ) ( -1 ) ( -1 ) ( -1 ) ( -1 ) ( -1 ) ( -1 ) ( -1 ) ).
    lv_result = z_humaneval_142=>sum_squares( lt_input ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = -3 ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: lt_input  TYPE TABLE OF i,
          lv_result TYPE i.

    lt_input = VALUE #( ( 0 ) ).
    lv_result = z_humaneval_142=>sum_squares( lt_input ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 0 ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: lt_input  TYPE TABLE OF i,
          lv_result TYPE i.

    lt_input = VALUE #( ( -1 ) ( -5 ) ( 2 ) ( -1 ) ( -5 ) ).
    lv_result = z_humaneval_142=>sum_squares( lt_input ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = -126 ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: lt_input  TYPE TABLE OF i,
          lv_result TYPE i.

    lt_input = VALUE #( ( -56 ) ( -99 ) ( 1 ) ( 0 ) ( -2 ) ).
    lv_result = z_humaneval_142=>sum_squares( lt_input ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 3030 ).
  ENDMETHOD.

  METHOD test_case_9.
    DATA: lt_input  TYPE TABLE OF i,
          lv_result TYPE i.

    lt_input = VALUE #( ( -1 ) ( 0 ) ( 0 ) ( 0 ) ( 0 ) ( 0 ) ( 0 ) ( 0 ) ( -1 ) ).
    lv_result = z_humaneval_142=>sum_squares( lt_input ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 0 ).
  ENDMETHOD.

  METHOD test_case_10.
    DATA: lt_input  TYPE TABLE OF i,
          lv_result TYPE i.

    lt_input = VALUE #( ( -16 ) ( -9 ) ( -2 ) ( 36 ) ( 36 ) ( 26 ) ( -20 ) ( 25 ) ( -40 ) ( 20 ) ( -4 ) ( 12 ) ( -26 ) ( 35 ) ( 37 ) ).
    lv_result = z_humaneval_142=>sum_squares( lt_input ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = -14196 ).
  ENDMETHOD.

  METHOD test_case_11.
    DATA: lt_input  TYPE TABLE OF i,
          lv_result TYPE i.

    lt_input = VALUE #( ( -1 ) ( -3 ) ( 17 ) ( -1 ) ( -15 ) ( 13 ) ( -1 ) ( 14 ) ( -14 ) ( -12 ) ( -5 ) ( 14 ) ( -14 ) ( 6 ) ( 13 ) ( 11 ) ( 16 ) ( 16 ) ( 4 ) ( 10 ) ).
    lv_result = z_humaneval_142=>sum_squares( lt_input ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = -1448 ).
  ENDMETHOD.

ENDCLASS.