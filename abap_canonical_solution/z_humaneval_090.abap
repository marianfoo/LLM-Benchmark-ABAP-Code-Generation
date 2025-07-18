CLASS z_humaneval_090 DEFINITION PUBLIC.
  PUBLIC SECTION.
    CLASS-METHODS next_smallest
      IMPORTING lst           TYPE int4_table
      RETURNING VALUE(result) TYPE REF TO i.
ENDCLASS.

CLASS z_humaneval_090 IMPLEMENTATION.
  METHOD next_smallest.
    TYPES: int4_set TYPE SORTED TABLE OF i WITH UNIQUE KEY table_line.
    DATA(set_lst) = VALUE int4_set( ).

    LOOP AT lst INTO DATA(val).
      INSERT val INTO TABLE set_lst.
    ENDLOOP.

    DATA(sorted_lst) = VALUE int4_table( ).
    LOOP AT set_lst INTO val.
      INSERT val INTO TABLE sorted_lst.
    ENDLOOP.

    SORT sorted_lst ASCENDING.

    IF lines( sorted_lst ) < 2.
      CLEAR result.
    ELSE.
      DATA(second_val) = sorted_lst[ 2 ].
      CREATE DATA result.
      result->* = second_val.
    ENDIF.
  ENDMETHOD.
ENDCLASS.
