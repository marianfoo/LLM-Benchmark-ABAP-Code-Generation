CLASS z_humaneval_087 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    TYPES: BEGIN OF ty_coordinate,
             row TYPE i,
             col TYPE i,
           END OF ty_coordinate.

    TYPES: tt_coordinates TYPE STANDARD TABLE OF ty_coordinate WITH NON-UNIQUE DEFAULT KEY,
           tt_matrix      TYPE STANDARD TABLE OF int4_table WITH NON-UNIQUE DEFAULT KEY.

    CLASS-METHODS: check
      IMPORTING
        lst                   TYPE tt_matrix
        x                     TYPE i
      RETURNING
        VALUE(rt_coordinates) TYPE tt_coordinates.

  PRIVATE SECTION.

ENDCLASS.

CLASS z_humaneval_087 IMPLEMENTATION.

  METHOD check.
    DATA: lv_row_index  TYPE i,
          lv_col_index  TYPE i,
          ls_coordinate TYPE ty_coordinate,
          lt_row_coords TYPE tt_coordinates.

    CLEAR: rt_coordinates.

    LOOP AT lst INTO DATA(lt_current_row).
      lv_row_index = sy-tabix - 1.
      CLEAR: lt_row_coords.

      LOOP AT lt_current_row INTO DATA(lv_current_value).
        lv_col_index = sy-tabix - 1.

        IF lv_current_value = x.
          ls_coordinate-row = lv_row_index.
          ls_coordinate-col = lv_col_index.
          APPEND ls_coordinate TO lt_row_coords.
        ENDIF.
      ENDLOOP.

      SORT lt_row_coords BY col DESCENDING.

      APPEND LINES OF lt_row_coords TO rt_coordinates.
    ENDLOOP.

  ENDMETHOD.

ENDCLASS.