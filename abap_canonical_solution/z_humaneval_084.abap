CLASS z_humaneval_084 DEFINITION PUBLIC.
  PUBLIC SECTION.
    CLASS-METHODS:
      solve
        IMPORTING
          VALUE(n)      TYPE i
        RETURNING
          VALUE(result) TYPE string.
ENDCLASS.

CLASS z_humaneval_084 IMPLEMENTATION.

  METHOD solve.

    DATA: lv_remainder  TYPE i,
          lv_string     TYPE string,
          lv_char       TYPE c,
          lv_char_index TYPE i,
          lv_digit      TYPE i,
          lv_sum        TYPE i.

    lv_string = n.

    DO strlen( lv_string ) TIMES.
      lv_char_index = sy-index - 1.
      lv_char  = lv_string+lv_char_index(1).
      lv_digit = lv_char - '0'.
      lv_sum = lv_sum + lv_digit.
    ENDDO.

    WHILE lv_sum > 0.
      lv_remainder = lv_sum MOD 2.
      result = |{ lv_remainder }{ result }|.
      lv_sum = lv_sum DIV 2.
    ENDWHILE.



  ENDMETHOD.

ENDCLASS.
