CLASS z_humaneval_057 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS monotonic
      IMPORTING
        VALUE(l)      TYPE int4_table
      RETURNING
        VALUE(result) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_057 IMPLEMENTATION.

  METHOD monotonic.
    DATA: sorted_asc  TYPE int4_table,
          sorted_desc TYPE int4_table.

    sorted_asc = l.
    SORT sorted_asc BY table_line ASCENDING.

    sorted_desc = l.
    SORT sorted_desc BY table_line DESCENDING.

    IF l = sorted_asc OR l = sorted_desc.
      result = abap_true.
    ELSE.
      result = abap_false.
    ENDIF.
  ENDMETHOD.

ENDCLASS.