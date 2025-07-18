CLASS lcl_test_z_humaneval_122 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING.

ENDCLASS.

CLASS lcl_test_z_humaneval_122 IMPLEMENTATION.

  METHOD test_case_1.
    DATA: lt_arr    TYPE TABLE OF i,
          lv_result TYPE i.

    lt_arr = VALUE #( ( 1 ) ( -2 ) ( -3 ) ( 41 ) ( 57 ) ( 76 ) ( 87 ) ( 88 ) ( 99 ) ).

    lv_result = z_humaneval_122=>add_elements( arr = lt_arr k = 3 ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = -4
    ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: lt_arr    TYPE TABLE OF i,
          lv_result TYPE i.

    lt_arr = VALUE #( ( 111 ) ( 121 ) ( 3 ) ( 4000 ) ( 5 ) ( 6 ) ).

    lv_result = z_humaneval_122=>add_elements( arr = lt_arr k = 2 ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 0
    ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: lt_arr    TYPE TABLE OF i,
          lv_result TYPE i.

    lt_arr = VALUE #( ( 11 ) ( 21 ) ( 3 ) ( 90 ) ( 5 ) ( 6 ) ( 7 ) ( 8 ) ( 9 ) ).

    lv_result = z_humaneval_122=>add_elements( arr = lt_arr k = 4 ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 125
    ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: lt_arr    TYPE TABLE OF i,
          lv_result TYPE i.

    lt_arr = VALUE #( ( 111 ) ( 21 ) ( 3 ) ( 4000 ) ( 5 ) ( 6 ) ( 7 ) ( 8 ) ( 9 ) ).

    lv_result = z_humaneval_122=>add_elements( arr = lt_arr k = 4 ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 24
    ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: lt_arr    TYPE TABLE OF i,
          lv_result TYPE i.

    lt_arr = VALUE #( ( 1 ) ).

    lv_result = z_humaneval_122=>add_elements( arr = lt_arr k = 1 ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 1
    ).
  ENDMETHOD.

ENDCLASS.