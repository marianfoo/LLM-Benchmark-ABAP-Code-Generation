CLASS ltc_test_median DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING.

ENDCLASS.

CLASS ltc_test_median IMPLEMENTATION.

  METHOD test_case_1.
    DATA: lt_input    TYPE TABLE OF i,
          lv_result   TYPE f,
          lv_expected TYPE f VALUE 3.

    lt_input = VALUE #( ( 3 ) ( 1 ) ( 2 ) ( 4 ) ( 5 ) ).
    lv_result = z_humaneval_047=>median( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = lv_expected ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: lt_input    TYPE TABLE OF i,
          lv_result   TYPE f,
          lv_expected TYPE f VALUE '8.0'.

    lt_input = VALUE #( ( -10 ) ( 4 ) ( 6 ) ( 1000 ) ( 10 ) ( 20 ) ).
    lv_result = z_humaneval_047=>median( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = lv_expected ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: lt_input    TYPE TABLE OF i,
          lv_result   TYPE f,
          lv_expected TYPE f VALUE 5.

    lt_input = VALUE #( ( 5 ) ).
    lv_result = z_humaneval_047=>median( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = lv_expected ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: lt_input    TYPE TABLE OF i,
          lv_result   TYPE f,
          lv_expected TYPE f VALUE '5.5'.

    lt_input = VALUE #( ( 6 ) ( 5 ) ).
    lv_result = z_humaneval_047=>median( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = lv_expected ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: lt_input    TYPE TABLE OF i,
          lv_result   TYPE f,
          lv_expected TYPE f VALUE 7.

    lt_input = VALUE #( ( 8 ) ( 1 ) ( 3 ) ( 9 ) ( 9 ) ( 2 ) ( 7 ) ).
    lv_result = z_humaneval_047=>median( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = lv_expected ).
  ENDMETHOD.

ENDCLASS.