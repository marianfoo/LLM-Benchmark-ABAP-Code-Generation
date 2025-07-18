CLASS ltcl_test_max_fill DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    TYPES: BEGIN OF t_grid_row,
             row TYPE int4_table,
           END OF t_grid_row,
           t_grid TYPE TABLE OF t_grid_row.

    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_max_fill IMPLEMENTATION.
  METHOD test_case_1.
    DATA: grid TYPE t_grid.

    APPEND VALUE #( row = VALUE int4_table( ( 0 ) ( 0 ) ( 1 ) ( 0 ) ) ) TO grid.
    APPEND VALUE #( row = VALUE int4_table( ( 0 ) ( 1 ) ( 0 ) ( 0 ) ) ) TO grid.
    APPEND VALUE #( row = VALUE int4_table( ( 1 ) ( 1 ) ( 1 ) ( 1 ) ) ) TO grid.

    DATA(result) = z_humaneval_115=>max_fill( grid = grid capacity = 1 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 6 ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: grid TYPE t_grid.

    APPEND VALUE #( row = VALUE int4_table( ( 0 ) ( 0 ) ( 1 ) ( 1 ) ) ) TO grid.
    APPEND VALUE #( row = VALUE int4_table( ( 0 ) ( 0 ) ( 0 ) ( 0 ) ) ) TO grid.
    APPEND VALUE #( row = VALUE int4_table( ( 1 ) ( 1 ) ( 1 ) ( 1 ) ) ) TO grid.
    APPEND VALUE #( row = VALUE int4_table( ( 0 ) ( 1 ) ( 1 ) ( 1 ) ) ) TO grid.

    DATA(result) = z_humaneval_115=>max_fill( grid = grid capacity = 2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 5 ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: grid TYPE t_grid.

    APPEND VALUE #( row = VALUE int4_table( ( 0 ) ( 0 ) ( 0 ) ) ) TO grid.
    APPEND VALUE #( row = VALUE int4_table( ( 0 ) ( 0 ) ( 0 ) ) ) TO grid.

    DATA(result) = z_humaneval_115=>max_fill( grid = grid capacity = 5 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 0 ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: grid TYPE t_grid.

    APPEND VALUE #( row = VALUE int4_table( ( 1 ) ( 1 ) ( 1 ) ( 1 ) ) ) TO grid.
    APPEND VALUE #( row = VALUE int4_table( ( 1 ) ( 1 ) ( 1 ) ( 1 ) ) ) TO grid.

    DATA(result) = z_humaneval_115=>max_fill( grid = grid capacity = 2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 4 ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: grid TYPE t_grid.

    APPEND VALUE #( row = VALUE int4_table( ( 1 ) ( 1 ) ( 1 ) ( 1 ) ) ) TO grid.
    APPEND VALUE #( row = VALUE int4_table( ( 1 ) ( 1 ) ( 1 ) ( 1 ) ) ) TO grid.

    DATA(result) = z_humaneval_115=>max_fill( grid = grid capacity = 9 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 2 ).
  ENDMETHOD.
ENDCLASS.