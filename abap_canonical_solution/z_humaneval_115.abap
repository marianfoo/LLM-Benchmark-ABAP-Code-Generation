CLASS z_humaneval_115 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    TYPES: BEGIN OF t_grid_row,
             row TYPE int4_table,
           END OF t_grid_row,
           t_grid TYPE TABLE OF t_grid_row.

    CLASS-METHODS: max_fill
      IMPORTING
        VALUE(grid)     TYPE t_grid
        VALUE(capacity) TYPE i
      RETURNING
        VALUE(result)   TYPE i.
ENDCLASS.

CLASS z_humaneval_115 IMPLEMENTATION.
  METHOD max_fill.
    DATA: row_sum    TYPE i,
          buckets    TYPE i,
          grid_row   TYPE t_grid_row,
          cell_value TYPE i.

    result = 0.

    LOOP AT grid INTO grid_row.
      CLEAR row_sum.
      LOOP AT grid_row-row INTO cell_value.
        row_sum = row_sum + cell_value.
      ENDLOOP.

      IF row_sum > 0.
        buckets = row_sum DIV capacity.
        IF row_sum MOD capacity > 0.
          buckets = buckets + 1.
        ENDIF.
        result = result + buckets.
      ENDIF.
    ENDLOOP.
  ENDMETHOD.
ENDCLASS.