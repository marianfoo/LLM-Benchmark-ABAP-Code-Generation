CLASS z_humaneval_058 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: common
      IMPORTING
        l1            TYPE int4_table
        l2            TYPE int4_table
      RETURNING
        VALUE(result) TYPE int4_table.

  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_058 IMPLEMENTATION.
  METHOD common.
    DATA: temp_table  TYPE int4_table,
          temp_sorted TYPE int4_table.

    LOOP AT l1 INTO DATA(e1).
      LOOP AT l2 INTO DATA(e2).
        IF e1 = e2.
          READ TABLE temp_table TRANSPORTING NO FIELDS WITH KEY table_line = e1.
          IF sy-subrc <> 0.
            temp_table = VALUE #( BASE temp_table ( e1 ) ).
          ENDIF.
        ENDIF.
      ENDLOOP.
    ENDLOOP.

    temp_sorted = temp_table.
    SORT temp_sorted.
    result = temp_sorted.
  ENDMETHOD.
ENDCLASS.