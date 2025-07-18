CLASS z_humaneval_030 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS get_positive
      IMPORTING
        l             TYPE int4_table
      RETURNING
        VALUE(result) TYPE int4_table.

  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_030 IMPLEMENTATION.
  METHOD get_positive.
    LOOP AT l INTO DATA(e).
      IF e > 0.
        APPEND e TO result.
      ENDIF.
    ENDLOOP.
  ENDMETHOD.
ENDCLASS.