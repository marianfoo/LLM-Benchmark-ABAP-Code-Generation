CLASS z_humaneval_093 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: encode
      IMPORTING
        iv_message       TYPE string
      RETURNING
        VALUE(rv_result) TYPE string.

ENDCLASS.

CLASS z_humaneval_093 IMPLEMENTATION.

  METHOD encode.
    DATA: lv_char     TYPE c LENGTH 1,
          lv_new_char TYPE c LENGTH 1,
          lv_len      TYPE i,
          lv_index    TYPE i.

    rv_result = ''.
    lv_len = strlen( iv_message ).

    DO lv_len TIMES.
      lv_index = sy-index - 1.
      lv_char = iv_message+lv_index(1).

      CASE lv_char.
        WHEN 'a'. lv_new_char = 'A'.
        WHEN 'b'. lv_new_char = 'B'.
        WHEN 'c'. lv_new_char = 'C'.
        WHEN 'd'. lv_new_char = 'D'.
        WHEN 'e'. lv_new_char = 'E'.
        WHEN 'f'. lv_new_char = 'F'.
        WHEN 'g'. lv_new_char = 'G'.
        WHEN 'h'. lv_new_char = 'H'.
        WHEN 'i'. lv_new_char = 'I'.
        WHEN 'j'. lv_new_char = 'J'.
        WHEN 'k'. lv_new_char = 'K'.
        WHEN 'l'. lv_new_char = 'L'.
        WHEN 'm'. lv_new_char = 'M'.
        WHEN 'n'. lv_new_char = 'N'.
        WHEN 'o'. lv_new_char = 'O'.
        WHEN 'p'. lv_new_char = 'P'.
        WHEN 'q'. lv_new_char = 'Q'.
        WHEN 'r'. lv_new_char = 'R'.
        WHEN 's'. lv_new_char = 'S'.
        WHEN 't'. lv_new_char = 'T'.
        WHEN 'u'. lv_new_char = 'U'.
        WHEN 'v'. lv_new_char = 'V'.
        WHEN 'w'. lv_new_char = 'W'.
        WHEN 'x'. lv_new_char = 'X'.
        WHEN 'y'. lv_new_char = 'Y'.
        WHEN 'z'. lv_new_char = 'Z'.
        WHEN 'A'. lv_new_char = 'a'.
        WHEN 'B'. lv_new_char = 'b'.
        WHEN 'C'. lv_new_char = 'c'.
        WHEN 'D'. lv_new_char = 'd'.
        WHEN 'E'. lv_new_char = 'e'.
        WHEN 'F'. lv_new_char = 'f'.
        WHEN 'G'. lv_new_char = 'g'.
        WHEN 'H'. lv_new_char = 'h'.
        WHEN 'I'. lv_new_char = 'i'.
        WHEN 'J'. lv_new_char = 'j'.
        WHEN 'K'. lv_new_char = 'k'.
        WHEN 'L'. lv_new_char = 'l'.
        WHEN 'M'. lv_new_char = 'm'.
        WHEN 'N'. lv_new_char = 'n'.
        WHEN 'O'. lv_new_char = 'o'.
        WHEN 'P'. lv_new_char = 'p'.
        WHEN 'Q'. lv_new_char = 'q'.
        WHEN 'R'. lv_new_char = 'r'.
        WHEN 'S'. lv_new_char = 's'.
        WHEN 'T'. lv_new_char = 't'.
        WHEN 'U'. lv_new_char = 'u'.
        WHEN 'V'. lv_new_char = 'v'.
        WHEN 'W'. lv_new_char = 'w'.
        WHEN 'X'. lv_new_char = 'x'.
        WHEN 'Y'. lv_new_char = 'y'.
        WHEN 'Z'. lv_new_char = 'z'.
        WHEN OTHERS. lv_new_char = lv_char.
      ENDCASE.

      CASE lv_new_char.
        WHEN 'a'. lv_new_char = 'c'.
        WHEN 'e'. lv_new_char = 'g'.
        WHEN 'i'. lv_new_char = 'k'.
        WHEN 'o'. lv_new_char = 'q'.
        WHEN 'u'. lv_new_char = 'w'.
        WHEN 'A'. lv_new_char = 'C'.
        WHEN 'E'. lv_new_char = 'G'.
        WHEN 'I'. lv_new_char = 'K'.
        WHEN 'O'. lv_new_char = 'Q'.
        WHEN 'U'. lv_new_char = 'W'.
      ENDCASE.

      CONCATENATE rv_result lv_new_char INTO rv_result RESPECTING BLANKS.
    ENDDO.

  ENDMETHOD.

ENDCLASS.