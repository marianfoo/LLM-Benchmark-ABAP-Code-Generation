CLASS ltcl_test_humaneval_121 DEFINITION
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
      test_case_7 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_humaneval_121 IMPLEMENTATION.
  METHOD test_case_1.
    DATA: lt_input    TYPE TABLE OF i,
          lv_result   TYPE i,
          lv_expected TYPE i VALUE 12.

    lt_input = VALUE #( ( 5 ) ( 8 ) ( 7 ) ( 1 ) ).

    lv_result = z_humaneval_121=>solution( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = lv_expected ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: lt_input    TYPE TABLE OF i,
          lv_result   TYPE i,
          lv_expected TYPE i VALUE 9.

    lt_input = VALUE #( ( 3 ) ( 3 ) ( 3 ) ( 3 ) ( 3 ) ).

    lv_result = z_humaneval_121=>solution( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = lv_expected ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: lt_input    TYPE TABLE OF i,
          lv_result   TYPE i,
          lv_expected TYPE i VALUE 0.

    lt_input = VALUE #( ( 30 ) ( 13 ) ( 24 ) ( 321 ) ).

    lv_result = z_humaneval_121=>solution( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = lv_expected ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: lt_input    TYPE TABLE OF i,
          lv_result   TYPE i,
          lv_expected TYPE i VALUE 5.

    lt_input = VALUE #( ( 5 ) ( 9 ) ).

    lv_result = z_humaneval_121=>solution( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = lv_expected ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: lt_input    TYPE TABLE OF i,
          lv_result   TYPE i,
          lv_expected TYPE i VALUE 0.

    lt_input = VALUE #( ( 2 ) ( 4 ) ( 8 ) ).

    lv_result = z_humaneval_121=>solution( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = lv_expected ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: lt_input    TYPE TABLE OF i,
          lv_result   TYPE i,
          lv_expected TYPE i VALUE 23.

    lt_input = VALUE #( ( 30 ) ( 13 ) ( 23 ) ( 32 ) ).

    lv_result = z_humaneval_121=>solution( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = lv_expected ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: lt_input    TYPE TABLE OF i,
          lv_result   TYPE i,
          lv_expected TYPE i VALUE 3.

    lt_input = VALUE #( ( 3 ) ( 13 ) ( 2 ) ( 9 ) ).

    lv_result = z_humaneval_121=>solution( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = lv_expected ).
  ENDMETHOD.
ENDCLASS.