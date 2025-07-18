CLASS z_humaneval_088_ut DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    METHODS:
      test_empty_array FOR TESTING,
      test_single_element FOR TESTING,
      test_example_1 FOR TESTING,
      test_example_2 FOR TESTING,
      test_edge_case_1 FOR TESTING,
      test_edge_case_2 FOR TESTING,
      test_edge_case_3 FOR TESTING.
ENDCLASS.

CLASS z_humaneval_088_ut IMPLEMENTATION.

  METHOD test_empty_array.
    DATA result TYPE int4_table.

    result = z_humaneval_088=>sort_array( VALUE #( ) ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = VALUE int4_table( )
      msg = 'Error: empty array' ).
  ENDMETHOD.

  METHOD test_single_element.
    DATA result TYPE int4_table.

    result = z_humaneval_088=>sort_array( VALUE #( ( 5 ) ) ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = VALUE int4_table( ( 5 ) )
      msg = 'Error: single element' ).
  ENDMETHOD.

  METHOD test_example_1.
    DATA result TYPE int4_table.

    result = z_humaneval_088=>sort_array( VALUE #( ( 2 ) ( 4 ) ( 3 ) ( 0 ) ( 1 ) ( 5 ) ) ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = VALUE int4_table( ( 0 ) ( 1 ) ( 2 ) ( 3 ) ( 4 ) ( 5 ) )
      msg = 'Error: example 1' ).
  ENDMETHOD.

  METHOD test_example_2.
    DATA result TYPE int4_table.

    result = z_humaneval_088=>sort_array( VALUE #( ( 2 ) ( 4 ) ( 3 ) ( 0 ) ( 1 ) ( 5 ) ( 6 ) ) ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = VALUE int4_table( ( 6 ) ( 5 ) ( 4 ) ( 3 ) ( 2 ) ( 1 ) ( 0 ) )
      msg = 'Error: example 2' ).
  ENDMETHOD.

  METHOD test_edge_case_1.
    DATA result TYPE int4_table.

    result = z_humaneval_088=>sort_array( VALUE #( ( 2 ) ( 1 ) ) ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = VALUE int4_table( ( 1 ) ( 2 ) )
      msg = 'Error: edge case 1' ).
  ENDMETHOD.

  METHOD test_edge_case_2.
    DATA result TYPE int4_table.

    result = z_humaneval_088=>sort_array( VALUE #( ( 15 ) ( 42 ) ( 87 ) ( 32 ) ( 11 ) ( 0 ) ) ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = VALUE int4_table( ( 0 ) ( 11 ) ( 15 ) ( 32 ) ( 42 ) ( 87 ) )
      msg = 'Error: edge case 2' ).
  ENDMETHOD.

  METHOD test_edge_case_3.
    DATA result TYPE int4_table.

    result = z_humaneval_088=>sort_array( VALUE #( ( 21 ) ( 14 ) ( 23 ) ( 11 ) ) ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = VALUE int4_table( ( 23 ) ( 21 ) ( 14 ) ( 11 ) )
      msg = 'Error: edge case 3' ).
  ENDMETHOD.

ENDCLASS.