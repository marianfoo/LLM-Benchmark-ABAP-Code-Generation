CLASS z_humaneval_034 DEFINITION
  PUBLIC
  FINAL
  ABSTRACT.

  PUBLIC SECTION.
    CLASS-METHODS unique
      IMPORTING
        VALUE(l)   TYPE int4_table
      RETURNING
        VALUE(rt_unique) TYPE int4_table.

ENDCLASS.

CLASS z_humaneval_034 IMPLEMENTATION.

  METHOD unique.
    DATA: lt_unique TYPE int4_table.

    " Remove duplicates while preserving order
    LOOP AT l INTO DATA(lv_value).
      READ TABLE lt_unique TRANSPORTING NO FIELDS WITH KEY table_line = lv_value.
      IF sy-subrc <> 0.
        APPEND lv_value TO lt_unique.
      ENDIF.
    ENDLOOP.

    " Sort the unique values
    SORT lt_unique.

    rt_unique = lt_unique.
  ENDMETHOD.

ENDCLASS.