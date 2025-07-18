CLASS ltc_test_z_humaneval_108 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    TYPES: BEGIN OF ty_int_table,
             value TYPE i,
           END OF ty_int_table.
    TYPES: tt_int_table TYPE TABLE OF ty_int_table.
    METHODS: test_empty_array FOR TESTING,
      test_negative_and_zero FOR TESTING,
      test_mixed_array_1 FOR TESTING,
      test_mixed_array_2 FOR TESTING,
      test_mixed_array_3 FOR TESTING,
      test_mixed_array_4 FOR TESTING,
      test_zero_and_one FOR TESTING,
      test_single_element FOR TESTING.

ENDCLASS.

CLASS ltc_test_z_humaneval_108 IMPLEMENTATION.

  METHOD test_empty_array.
    DATA: lt_input  TYPE tt_int_table,
          lv_result TYPE i.

    lv_result = z_humaneval_108=>count_nums( lt_input ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 0 ).
  ENDMETHOD.

  METHOD test_negative_and_zero.
    DATA: lt_input  TYPE tt_int_table,
          lv_result TYPE i.

    APPEND VALUE #( value = -1 ) TO lt_input.
    APPEND VALUE #( value = -2 ) TO lt_input.
    APPEND VALUE #( value = 0 ) TO lt_input.

    lv_result = z_humaneval_108=>count_nums( lt_input ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 0 ).
  ENDMETHOD.

  METHOD test_mixed_array_1.
    DATA: lt_input  TYPE tt_int_table,
          lv_result TYPE i.

    APPEND VALUE #( value = 1 ) TO lt_input.
    APPEND VALUE #( value = 1 ) TO lt_input.
    APPEND VALUE #( value = 2 ) TO lt_input.
    APPEND VALUE #( value = -2 ) TO lt_input.
    APPEND VALUE #( value = 3 ) TO lt_input.
    APPEND VALUE #( value = 4 ) TO lt_input.
    APPEND VALUE #( value = 5 ) TO lt_input.

    lv_result = z_humaneval_108=>count_nums( lt_input ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 6 ).
  ENDMETHOD.

  METHOD test_mixed_array_2.
    DATA: lt_input  TYPE tt_int_table,
          lv_result TYPE i.

    APPEND VALUE #( value = 1 ) TO lt_input.
    APPEND VALUE #( value = 6 ) TO lt_input.
    APPEND VALUE #( value = 9 ) TO lt_input.
    APPEND VALUE #( value = -6 ) TO lt_input.
    APPEND VALUE #( value = 0 ) TO lt_input.
    APPEND VALUE #( value = 1 ) TO lt_input.
    APPEND VALUE #( value = 5 ) TO lt_input.

    lv_result = z_humaneval_108=>count_nums( lt_input ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 5 ).
  ENDMETHOD.

  METHOD test_mixed_array_3.
    DATA: lt_input  TYPE tt_int_table,
          lv_result TYPE i.

    APPEND VALUE #( value = 1 ) TO lt_input.
    APPEND VALUE #( value = 100 ) TO lt_input.
    APPEND VALUE #( value = 98 ) TO lt_input.
    APPEND VALUE #( value = -7 ) TO lt_input.
    APPEND VALUE #( value = 1 ) TO lt_input.
    APPEND VALUE #( value = -1 ) TO lt_input.

    lv_result = z_humaneval_108=>count_nums( lt_input ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 4 ).
  ENDMETHOD.

  METHOD test_mixed_array_4.
    DATA: lt_input  TYPE tt_int_table,
          lv_result TYPE i.

    APPEND VALUE #( value = 12 ) TO lt_input.
    APPEND VALUE #( value = 23 ) TO lt_input.
    APPEND VALUE #( value = 34 ) TO lt_input.
    APPEND VALUE #( value = -45 ) TO lt_input.
    APPEND VALUE #( value = -56 ) TO lt_input.
    APPEND VALUE #( value = 0 ) TO lt_input.

    lv_result = z_humaneval_108=>count_nums( lt_input ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 5 ).
  ENDMETHOD.

  METHOD test_zero_and_one.
    DATA: lt_input  TYPE tt_int_table,
          lv_result TYPE i.

    APPEND VALUE #( value = 0 ) TO lt_input.
    APPEND VALUE #( value = 1 ) TO lt_input.

    lv_result = z_humaneval_108=>count_nums( lt_input ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 1 ).
  ENDMETHOD.

  METHOD test_single_element.
    DATA: lt_input  TYPE tt_int_table,
          lv_result TYPE i.

    APPEND VALUE #( value = 1 ) TO lt_input.

    lv_result = z_humaneval_108=>count_nums( lt_input ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 1 ).
  ENDMETHOD.

ENDCLASS.