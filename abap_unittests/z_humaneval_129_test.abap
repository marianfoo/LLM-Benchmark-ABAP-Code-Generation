CLASS z_humaneval_129_test DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    TYPES: BEGIN OF ty_grid_row,
             values TYPE int4_table,
           END OF ty_grid_row,
           ty_grid TYPE TABLE OF ty_grid_row.

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

CLASS z_humaneval_129_test IMPLEMENTATION.

  METHOD test_case_1.
    DATA: grid     TYPE ty_grid,
          result   TYPE int4_table,
          expected TYPE int4_table.

    grid = VALUE #( ( values = VALUE #( ( 1 ) ( 2 ) ( 3 ) ) )
                    ( values = VALUE #( ( 4 ) ( 5 ) ( 6 ) ) )
                    ( values = VALUE #( ( 7 ) ( 8 ) ( 9 ) ) ) ).
    expected = VALUE #( ( 1 ) ( 2 ) ( 1 ) ).

    result = z_humaneval_129=>minpath( grid = grid k = 3 ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: grid     TYPE ty_grid,
          result   TYPE int4_table,
          expected TYPE int4_table.

    grid = VALUE #( ( values = VALUE #( ( 5 ) ( 9 ) ( 3 ) ) )
                    ( values = VALUE #( ( 4 ) ( 1 ) ( 6 ) ) )
                    ( values = VALUE #( ( 7 ) ( 8 ) ( 2 ) ) ) ).
    expected = VALUE #( ( 1 ) ).

    result = z_humaneval_129=>minpath( grid = grid k = 1 ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: grid     TYPE ty_grid,
          result   TYPE int4_table,
          expected TYPE int4_table.

    grid = VALUE #( ( values = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ) )
                    ( values = VALUE #( ( 5 ) ( 6 ) ( 7 ) ( 8 ) ) )
                    ( values = VALUE #( ( 9 ) ( 10 ) ( 11 ) ( 12 ) ) )
                    ( values = VALUE #( ( 13 ) ( 14 ) ( 15 ) ( 16 ) ) ) ).
    expected = VALUE #( ( 1 ) ( 2 ) ( 1 ) ( 2 ) ).

    result = z_humaneval_129=>minpath( grid = grid k = 4 ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: grid     TYPE ty_grid,
          result   TYPE int4_table,
          expected TYPE int4_table.

    grid = VALUE #( ( values = VALUE #( ( 6 ) ( 4 ) ( 13 ) ( 10 ) ) )
                    ( values = VALUE #( ( 5 ) ( 7 ) ( 12 ) ( 1 ) ) )
                    ( values = VALUE #( ( 3 ) ( 16 ) ( 11 ) ( 15 ) ) )
                    ( values = VALUE #( ( 8 ) ( 14 ) ( 9 ) ( 2 ) ) ) ).
    expected = VALUE #( ( 1 ) ( 10 ) ( 1 ) ( 10 ) ( 1 ) ( 10 ) ( 1 ) ).

    result = z_humaneval_129=>minpath( grid = grid k = 7 ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: grid     TYPE ty_grid,
          result   TYPE int4_table,
          expected TYPE int4_table.

    grid = VALUE #( ( values = VALUE #( ( 8 ) ( 14 ) ( 9 ) ( 2 ) ) )
                    ( values = VALUE #( ( 6 ) ( 4 ) ( 13 ) ( 15 ) ) )
                    ( values = VALUE #( ( 5 ) ( 7 ) ( 1 ) ( 12 ) ) )
                    ( values = VALUE #( ( 3 ) ( 10 ) ( 11 ) ( 16 ) ) ) ).
    expected = VALUE #( ( 1 ) ( 7 ) ( 1 ) ( 7 ) ( 1 ) ).

    result = z_humaneval_129=>minpath( grid = grid k = 5 ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: grid     TYPE ty_grid,
          result   TYPE int4_table,
          expected TYPE int4_table.

    grid = VALUE #( ( values = VALUE #( ( 11 ) ( 8 ) ( 7 ) ( 2 ) ) )
                    ( values = VALUE #( ( 5 ) ( 16 ) ( 14 ) ( 4 ) ) )
                    ( values = VALUE #( ( 9 ) ( 3 ) ( 15 ) ( 6 ) ) )
                    ( values = VALUE #( ( 12 ) ( 13 ) ( 10 ) ( 1 ) ) ) ).
    expected = VALUE #( ( 1 ) ( 6 ) ( 1 ) ( 6 ) ( 1 ) ( 6 ) ( 1 ) ( 6 ) ( 1 ) ).

    result = z_humaneval_129=>minpath( grid = grid k = 9 ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: grid     TYPE ty_grid,
          result   TYPE int4_table,
          expected TYPE int4_table.

    grid = VALUE #( ( values = VALUE #( ( 12 ) ( 13 ) ( 10 ) ( 1 ) ) )
                    ( values = VALUE #( ( 9 ) ( 3 ) ( 15 ) ( 6 ) ) )
                    ( values = VALUE #( ( 5 ) ( 16 ) ( 14 ) ( 4 ) ) )
                    ( values = VALUE #( ( 11 ) ( 8 ) ( 7 ) ( 2 ) ) ) ).
    expected = VALUE #( ( 1 ) ( 6 ) ( 1 ) ( 6 ) ( 1 ) ( 6 ) ( 1 ) ( 6 ) ( 1 ) ( 6 ) ( 1 ) ( 6 ) ).

    result = z_humaneval_129=>minpath( grid = grid k = 12 ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: grid     TYPE ty_grid,
          result   TYPE int4_table,
          expected TYPE int4_table.

    grid = VALUE #( ( values = VALUE #( ( 2 ) ( 7 ) ( 4 ) ) )
                    ( values = VALUE #( ( 3 ) ( 1 ) ( 5 ) ) )
                    ( values = VALUE #( ( 6 ) ( 8 ) ( 9 ) ) ) ).
    expected = VALUE #( ( 1 ) ( 3 ) ( 1 ) ( 3 ) ( 1 ) ( 3 ) ( 1 ) ( 3 ) ).

    result = z_humaneval_129=>minpath( grid = grid k = 8 ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_case_9.
    DATA: grid     TYPE ty_grid,
          result   TYPE int4_table,
          expected TYPE int4_table.

    grid = VALUE #( ( values = VALUE #( ( 6 ) ( 1 ) ( 5 ) ) )
                    ( values = VALUE #( ( 3 ) ( 8 ) ( 9 ) ) )
                    ( values = VALUE #( ( 2 ) ( 7 ) ( 4 ) ) ) ).
    expected = VALUE #( ( 1 ) ( 5 ) ( 1 ) ( 5 ) ( 1 ) ( 5 ) ( 1 ) ( 5 ) ).

    result = z_humaneval_129=>minpath( grid = grid k = 8 ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_case_10.
    DATA: grid     TYPE ty_grid,
          result   TYPE int4_table,
          expected TYPE int4_table.

    grid = VALUE #( ( values = VALUE #( ( 1 ) ( 2 ) ) )
                    ( values = VALUE #( ( 3 ) ( 4 ) ) ) ).
    expected = VALUE #( ( 1 ) ( 2 ) ( 1 ) ( 2 ) ( 1 ) ( 2 ) ( 1 ) ( 2 ) ( 1 ) ( 2 ) ).

    result = z_humaneval_129=>minpath( grid = grid k = 10 ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_case_11.
    DATA: grid     TYPE ty_grid,
          result   TYPE int4_table,
          expected TYPE int4_table.

    grid = VALUE #( ( values = VALUE #( ( 1 ) ( 3 ) ) )
                    ( values = VALUE #( ( 3 ) ( 2 ) ) ) ).
    expected = VALUE #( ( 1 ) ( 3 ) ( 1 ) ( 3 ) ( 1 ) ( 3 ) ( 1 ) ( 3 ) ( 1 ) ( 3 ) ).

    result = z_humaneval_129=>minpath( grid = grid k = 10 ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

ENDCLASS.