CLASS ltcl_test_z_humaneval_087 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    TYPES: BEGIN OF ty_coordinate,
             row TYPE i,
             col TYPE i,
           END OF ty_coordinate.

    TYPES: tt_coordinates TYPE STANDARD TABLE OF ty_coordinate WITH NON-UNIQUE DEFAULT KEY,
           tt_matrix      TYPE STANDARD TABLE OF int4_table WITH NON-UNIQUE DEFAULT KEY.

    METHODS: test_basic_case FOR TESTING,
      test_column_2 FOR TESTING,
      test_multiple_ones FOR TESTING,
      test_empty_matrix FOR TESTING,
      test_no_match FOR TESTING,
      test_edge_case FOR TESTING.

ENDCLASS.

CLASS ltcl_test_z_humaneval_087 IMPLEMENTATION.

  METHOD test_basic_case.
    " Test case: [[1,2,3,4,5,6],[1,2,3,4,1,6], [1,2,3,4,5,1]], 1
    " Expected: [(0, 0), (1, 4), (1, 0), (2, 5), (2, 0)]
    DATA: lt_matrix   TYPE tt_matrix,
          lt_row      TYPE int4_table,
          lt_result   TYPE tt_coordinates,
          lt_expected TYPE tt_coordinates,
          ls_coord    TYPE ty_coordinate.

    " Build matrix
    CLEAR: lt_row.
    APPEND: 1 TO lt_row, 2 TO lt_row, 3 TO lt_row, 4 TO lt_row, 5 TO lt_row, 6 TO lt_row.
    APPEND lt_row TO lt_matrix.

    CLEAR: lt_row.
    APPEND: 1 TO lt_row, 2 TO lt_row, 3 TO lt_row, 4 TO lt_row, 1 TO lt_row, 6 TO lt_row.
    APPEND lt_row TO lt_matrix.

    CLEAR: lt_row.
    APPEND: 1 TO lt_row, 2 TO lt_row, 3 TO lt_row, 4 TO lt_row, 5 TO lt_row, 1 TO lt_row.
    APPEND lt_row TO lt_matrix.

    " Build expected result
    ls_coord-row = 0. ls_coord-col = 0. APPEND ls_coord TO lt_expected.
    ls_coord-row = 1. ls_coord-col = 4. APPEND ls_coord TO lt_expected.
    ls_coord-row = 1. ls_coord-col = 0. APPEND ls_coord TO lt_expected.
    ls_coord-row = 2. ls_coord-col = 5. APPEND ls_coord TO lt_expected.
    ls_coord-row = 2. ls_coord-col = 0. APPEND ls_coord TO lt_expected.

    " Execute test
    lt_result = z_humaneval_087=>check( iv_matrix = lt_matrix iv_search_value = 1 ).

    " Assert
    cl_abap_unit_assert=>assert_equals( act = lt_result exp = lt_expected ).
  ENDMETHOD.

  METHOD test_column_2.
    " Test case: 6 rows of [1,2,3,4,5,6], search for 2
    " Expected: [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
    DATA: lt_matrix   TYPE tt_matrix,
          lt_row      TYPE int4_table,
          lt_result   TYPE tt_coordinates,
          lt_expected TYPE tt_coordinates,
          ls_coord    TYPE ty_coordinate,
          lv_i        TYPE i.

    DO 6 TIMES.
      CLEAR: lt_row.
      APPEND: 1 TO lt_row, 2 TO lt_row, 3 TO lt_row, 4 TO lt_row, 5 TO lt_row, 6 TO lt_row.
      APPEND lt_row TO lt_matrix.
    ENDDO.

    DO 6 TIMES.
      lv_i = sy-index - 1.
      ls_coord-row = lv_i. ls_coord-col = 1. APPEND ls_coord TO lt_expected.
    ENDDO.

    lt_result = z_humaneval_087=>check( iv_matrix = lt_matrix iv_search_value = 2 ).

    cl_abap_unit_assert=>assert_equals( act = lt_result exp = lt_expected ).
  ENDMETHOD.

  METHOD test_multiple_ones.
    " Test case with multiple 1s in different positions
    " Expected: [(0, 0), (1, 0), (2, 1), (2, 0), (3, 2), (3, 0), (4, 3), (4, 0), (5, 4), (5, 0), (6, 5), (6, 0)]
    DATA: lt_matrix   TYPE tt_matrix,
          lt_row      TYPE int4_table,
          lt_result   TYPE tt_coordinates,
          lt_expected TYPE tt_coordinates,
          ls_coord    TYPE ty_coordinate.

    CLEAR: lt_row.
    APPEND: 1 TO lt_row, 2 TO lt_row, 3 TO lt_row, 4 TO lt_row, 5 TO lt_row, 6 TO lt_row.
    APPEND lt_row TO lt_matrix.

    CLEAR: lt_row.
    APPEND: 1 TO lt_row, 2 TO lt_row, 3 TO lt_row, 4 TO lt_row, 5 TO lt_row, 6 TO lt_row.
    APPEND lt_row TO lt_matrix.

    CLEAR: lt_row.
    APPEND: 1 TO lt_row, 1 TO lt_row, 3 TO lt_row, 4 TO lt_row, 5 TO lt_row, 6 TO lt_row.
    APPEND lt_row TO lt_matrix.

    CLEAR: lt_row.
    APPEND: 1 TO lt_row, 2 TO lt_row, 1 TO lt_row, 4 TO lt_row, 5 TO lt_row, 6 TO lt_row.
    APPEND lt_row TO lt_matrix.

    CLEAR: lt_row.
    APPEND: 1 TO lt_row, 2 TO lt_row, 3 TO lt_row, 1 TO lt_row, 5 TO lt_row, 6 TO lt_row.
    APPEND lt_row TO lt_matrix.

    CLEAR: lt_row.
    APPEND: 1 TO lt_row, 2 TO lt_row, 3 TO lt_row, 4 TO lt_row, 1 TO lt_row, 6 TO lt_row.
    APPEND lt_row TO lt_matrix.

    CLEAR: lt_row.
    APPEND: 1 TO lt_row, 2 TO lt_row, 3 TO lt_row, 4 TO lt_row, 5 TO lt_row, 1 TO lt_row.
    APPEND lt_row TO lt_matrix.

    ls_coord-row = 0. ls_coord-col = 0. APPEND ls_coord TO lt_expected.
    ls_coord-row = 1. ls_coord-col = 0. APPEND ls_coord TO lt_expected.
    ls_coord-row = 2. ls_coord-col = 1. APPEND ls_coord TO lt_expected.
    ls_coord-row = 2. ls_coord-col = 0. APPEND ls_coord TO lt_expected.
    ls_coord-row = 3. ls_coord-col = 2. APPEND ls_coord TO lt_expected.
    ls_coord-row = 3. ls_coord-col = 0. APPEND ls_coord TO lt_expected.
    ls_coord-row = 4. ls_coord-col = 3. APPEND ls_coord TO lt_expected.
    ls_coord-row = 4. ls_coord-col = 0. APPEND ls_coord TO lt_expected.
    ls_coord-row = 5. ls_coord-col = 4. APPEND ls_coord TO lt_expected.
    ls_coord-row = 5. ls_coord-col = 0. APPEND ls_coord TO lt_expected.
    ls_coord-row = 6. ls_coord-col = 5. APPEND ls_coord TO lt_expected.
    ls_coord-row = 6. ls_coord-col = 0. APPEND ls_coord TO lt_expected.

    lt_result = z_humaneval_087=>check( iv_matrix = lt_matrix iv_search_value = 1 ).

    cl_abap_unit_assert=>assert_equals( act = lt_result exp = lt_expected ).
  ENDMETHOD.

  METHOD test_empty_matrix.
    " Test case: [], 1 - Expected: []
    DATA: lt_matrix   TYPE tt_matrix,
          lt_result   TYPE tt_coordinates,
          lt_expected TYPE tt_coordinates.

    lt_result = z_humaneval_087=>check( iv_matrix = lt_matrix iv_search_value = 1 ).

    cl_abap_unit_assert=>assert_equals( act = lt_result exp = lt_expected ).
  ENDMETHOD.

  METHOD test_no_match.
    " Test case: [[1]], 2 - Expected: []
    DATA: lt_matrix   TYPE tt_matrix,
          lt_row      TYPE int4_table,
          lt_result   TYPE tt_coordinates,
          lt_expected TYPE tt_coordinates.

    CLEAR: lt_row.
    APPEND 1 TO lt_row.
    APPEND lt_row TO lt_matrix.

    lt_result = z_humaneval_087=>check( iv_matrix = lt_matrix iv_search_value = 2 ).

    cl_abap_unit_assert=>assert_equals( act = lt_result exp = lt_expected ).
  ENDMETHOD.

  METHOD test_edge_case.
    " Test case: [[], [1], [1, 2, 3]], 3 - Expected: [(2, 2)]
    DATA: lt_matrix   TYPE tt_matrix,
          lt_row      TYPE int4_table,
          lt_result   TYPE tt_coordinates,
          lt_expected TYPE tt_coordinates,
          ls_coord    TYPE ty_coordinate.

    CLEAR: lt_row.
    APPEND lt_row TO lt_matrix.

    CLEAR: lt_row.
    APPEND 1 TO lt_row.
    APPEND lt_row TO lt_matrix.

    CLEAR: lt_row.
    APPEND: 1 TO lt_row, 2 TO lt_row, 3 TO lt_row.
    APPEND lt_row TO lt_matrix.

    ls_coord-row = 2. ls_coord-col = 2. APPEND ls_coord TO lt_expected.

    lt_result = z_humaneval_087=>check( iv_matrix = lt_matrix iv_search_value = 3 ).

    cl_abap_unit_assert=>assert_equals( act = lt_result exp = lt_expected ).
  ENDMETHOD.

ENDCLASS.