CLASS lhc_test DEFINITION
  FOR TESTING
  FINAL
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_empty_list FOR TESTING,
      test_increasing_order FOR TESTING,
      test_decreasing_order FOR TESTING,
      test_mixed_values FOR TESTING.
ENDCLASS.

CLASS lhc_test IMPLEMENTATION.

  METHOD test_empty_list.
    DATA: numbers  TYPE int4_table,
          expected TYPE int4_table.

    CLEAR: numbers, expected.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_009=>rolling_max( numbers )
      exp = expected ).
  ENDMETHOD.

  METHOD test_increasing_order.
    DATA: numbers  TYPE int4_table,
          expected TYPE int4_table.

    numbers  = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ).
    expected = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_009=>rolling_max( numbers )
      exp = expected ).
  ENDMETHOD.

  METHOD test_decreasing_order.
    DATA: numbers  TYPE int4_table,
          expected TYPE int4_table.

    numbers  = VALUE #( ( 4 ) ( 3 ) ( 2 ) ( 1 ) ).
    expected = VALUE #( ( 4 ) ( 4 ) ( 4 ) ( 4 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_009=>rolling_max( numbers )
      exp = expected ).
  ENDMETHOD.

  METHOD test_mixed_values.
    DATA: numbers  TYPE int4_table,
          expected TYPE int4_table.

    numbers  = VALUE #( ( 3 ) ( 2 ) ( 3 ) ( 100 ) ( 3 ) ).
    expected = VALUE #( ( 3 ) ( 3 ) ( 3 ) ( 100 ) ( 100 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_009=>rolling_max( numbers )
      exp = expected ).
  ENDMETHOD.

ENDCLASS.