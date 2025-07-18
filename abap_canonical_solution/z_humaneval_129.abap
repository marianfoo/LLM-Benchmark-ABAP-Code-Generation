CLASS z_humaneval_129 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    TYPES: BEGIN OF ty_grid_row,
             values TYPE int4_table,
           END OF ty_grid_row,
           ty_grid TYPE TABLE OF ty_grid_row.

    CLASS-METHODS: minpath
      IMPORTING
        grid          TYPE ty_grid
        k             TYPE i
      RETURNING
        VALUE(result) TYPE int4_table.

ENDCLASS.

CLASS z_humaneval_129 IMPLEMENTATION.

  METHOD minpath.
    DATA: n           TYPE i,
          val         TYPE i,
          temp        TYPE int4_table,
          i           TYPE i,
          j           TYPE i,
          idx         TYPE i,
          current_val TYPE i.

    n = lines( grid ).
    val = n * n + 1.

    DO n TIMES.
      i = sy-index - 1.
      READ TABLE grid INDEX ( i + 1 ) INTO DATA(row).
      DO n TIMES.
        j = sy-index - 1.
        READ TABLE row-values INDEX ( j + 1 ) INTO current_val.
        IF current_val = 1.
          CLEAR temp.
          IF i <> 0.
            READ TABLE grid INDEX i INTO DATA(prev_row).
            READ TABLE prev_row-values INDEX ( j + 1 ) INTO DATA(neighbor_val).
            temp = VALUE #( BASE temp ( neighbor_val ) ).
          ENDIF.
          IF j <> 0.
            READ TABLE row-values INDEX j INTO neighbor_val.
            temp = VALUE #( BASE temp ( neighbor_val ) ).
          ENDIF.
          IF i <> n - 1.
            READ TABLE grid INDEX ( i + 2 ) INTO DATA(next_row).
            READ TABLE next_row-values INDEX ( j + 1 ) INTO neighbor_val.
            temp = VALUE #( BASE temp ( neighbor_val ) ).
          ENDIF.
          IF j <> n - 1.
            READ TABLE row-values INDEX ( j + 2 ) INTO neighbor_val.
            temp = VALUE #( BASE temp ( neighbor_val ) ).
          ENDIF.
          LOOP AT temp INTO DATA(temp_val).
            IF temp_val < val.
              val = temp_val.
            ENDIF.
          ENDLOOP.
        ENDIF.
      ENDDO.
    ENDDO.

    DO k TIMES.
      idx = sy-index - 1.
      IF idx MOD 2 = 0.
        result = VALUE #( BASE result ( 1 ) ).
      ELSE.
        result = VALUE #( BASE result ( val ) ).
      ENDIF.
    ENDDO.

  ENDMETHOD.

ENDCLASS.