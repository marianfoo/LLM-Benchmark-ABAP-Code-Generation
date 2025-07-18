CLASS z_humaneval_027 DEFINITION
  PUBLIC
  FINAL.
  PUBLIC SECTION.
    CLASS-METHODS flip_case
      IMPORTING
        !string       TYPE string
      RETURNING
        VALUE(result) TYPE string.
ENDCLASS.

CLASS z_humaneval_027 IMPLEMENTATION.
  METHOD flip_case.
    DATA: lv_char  TYPE string,
          lv_len   TYPE i,
          lv_index TYPE i.

    lv_len = strlen( string ).
    result = ''.

    DO lv_len TIMES.
      lv_index = sy-index - 1.
      lv_char = string+lv_index(1).

      IF lv_char BETWEEN 'A' AND 'Z'.
        result = |{ result }{ to_lower( lv_char ) }|.
      ELSEIF lv_char BETWEEN 'a' AND 'z'.
        result = |{ result }{ to_upper( lv_char ) }|.
      ELSE.
        result = |{ result }{ lv_char }|.
      ENDIF.
    ENDDO.
  ENDMETHOD.
ENDCLASS.