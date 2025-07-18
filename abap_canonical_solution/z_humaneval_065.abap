CLASS z_humaneval_065 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: circular_shift
      IMPORTING
        VALUE(x)      TYPE i
        VALUE(shift)  TYPE i
      RETURNING
        VALUE(result) TYPE string.

ENDCLASS.

CLASS z_humaneval_065 IMPLEMENTATION.
  METHOD circular_shift.
    DATA: s     TYPE string,
          len_s TYPE i.

    s = |{ x }|.
    len_s = strlen( s ).

    IF shift > len_s.
      DATA: reversed TYPE string,
            i        TYPE i,
            offset   TYPE i.

      DO len_s TIMES.
        i = len_s - sy-index + 1.
        offset = i - 1.
        reversed = |{ reversed }{ s+offset(1) }|.
      ENDDO.

      result = reversed.
    ELSE.
      DATA: start_pos   TYPE i,
            first_part  TYPE string,
            second_part TYPE string.

      start_pos = len_s - shift.
      first_part = s+start_pos.
      second_part = s+0(start_pos).
      result = |{ first_part }{ second_part }|.
    ENDIF.
  ENDMETHOD.
ENDCLASS.