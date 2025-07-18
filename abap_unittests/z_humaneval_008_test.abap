CLASS lt_test_example DEFINITION
  FOR TESTING
  FINAL
  CREATE PUBLIC
  RISK LEVEL HARMLESS.

  PUBLIC SECTION.
    METHODS: test_empty_array FOR TESTING,
      test_three_ones FOR TESTING,
      test_with_zero FOR TESTING,
      test_three_numbers FOR TESTING,
      test_single_element FOR TESTING.
  PRIVATE SECTION.
ENDCLASS.

CLASS lt_test_example IMPLEMENTATION.

  METHOD test_empty_array.
    DATA(result) = z_humaneval_008=>sum_product( VALUE #( ) ).

    cl_abap_unit_assert=>assert_equals( act = result-sum_value exp = 0 ).
    cl_abap_unit_assert=>assert_equals( act = result-prod_value exp = 1 ).
  ENDMETHOD.

  METHOD test_three_ones.
    DATA(result) = z_humaneval_008=>sum_product( VALUE #( ( 1 ) ( 1 ) ( 1 ) ) ).

    cl_abap_unit_assert=>assert_equals( act = result-sum_value exp = 3 ).
    cl_abap_unit_assert=>assert_equals( act = result-prod_value exp = 1 ).
  ENDMETHOD.

  METHOD test_with_zero.
    DATA(result) = z_humaneval_008=>sum_product( VALUE #( ( 100 ) ( 0 ) ) ).

    cl_abap_unit_assert=>assert_equals( act = result-sum_value exp = 100 ).
    cl_abap_unit_assert=>assert_equals( act = result-prod_value exp = 0 ).
  ENDMETHOD.

  METHOD test_three_numbers.
    DATA(result) = z_humaneval_008=>sum_product( VALUE #( ( 3 ) ( 5 ) ( 7 ) ) ).

    cl_abap_unit_assert=>assert_equals( act = result-sum_value exp = 15 ).
    cl_abap_unit_assert=>assert_equals( act = result-prod_value exp = 105 ).
  ENDMETHOD.

  METHOD test_single_element.
    DATA(result) = z_humaneval_008=>sum_product( VALUE #( ( 10 ) ) ).

    cl_abap_unit_assert=>assert_equals( act = result-sum_value exp = 10 ).
    cl_abap_unit_assert=>assert_equals( act = result-prod_value exp = 10 ).
  ENDMETHOD.

ENDCLASS.