CLASS z_humaneval_011 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS string_xor
      IMPORTING
        !a            TYPE string
        !b            TYPE string
      RETURNING
        VALUE(result) TYPE string.
  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_011 IMPLEMENTATION.
  METHOD string_xor.
    DATA: lv_length TYPE i,
          lv_result TYPE string,
          lv_index  TYPE i,
          lv_char_a TYPE c LENGTH 1,
          lv_char_b TYPE c LENGTH 1.

    lv_length = strlen( a ).
    lv_result = ''.

    DO lv_length TIMES.
      lv_index = sy-index - 1.
      lv_char_a = a+lv_index.
      lv_char_b = b+lv_index.

      IF lv_char_a = lv_char_b.
        lv_result = lv_result && '0'.
      ELSE.
        lv_result = lv_result && '1'.
      ENDIF.
    ENDDO.

    result = lv_result.
  ENDMETHOD.
ENDCLASS.