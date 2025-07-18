CLASS z_humaneval_050 DEFINITION
PUBLIC
FINAL
CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: decode_shift
      IMPORTING
        VALUE(s)      TYPE string
      RETURNING
        VALUE(result) TYPE string.
  PROTECTED SECTION.
  PRIVATE SECTION.
    CLASS-METHODS: encode_shift
      IMPORTING
        VALUE(s)      TYPE string
      RETURNING
        VALUE(result) TYPE string.
ENDCLASS.

CLASS z_humaneval_050 IMPLEMENTATION.
  METHOD encode_shift.
    DATA: lv_char        TYPE c LENGTH 1,
          lv_result_char TYPE c LENGTH 1,
          lv_index       TYPE i.

    DO strlen( s ) TIMES.
      lv_index = sy-index - 1.
      lv_char = s+lv_index(1).

      CASE lv_char.
        WHEN 'a'. lv_result_char = 'f'.
        WHEN 'b'. lv_result_char = 'g'.
        WHEN 'c'. lv_result_char = 'h'.
        WHEN 'd'. lv_result_char = 'i'.
        WHEN 'e'. lv_result_char = 'j'.
        WHEN 'f'. lv_result_char = 'k'.
        WHEN 'g'. lv_result_char = 'l'.
        WHEN 'h'. lv_result_char = 'm'.
        WHEN 'i'. lv_result_char = 'n'.
        WHEN 'j'. lv_result_char = 'o'.
        WHEN 'k'. lv_result_char = 'p'.
        WHEN 'l'. lv_result_char = 'q'.
        WHEN 'm'. lv_result_char = 'r'.
        WHEN 'n'. lv_result_char = 's'.
        WHEN 'o'. lv_result_char = 't'.
        WHEN 'p'. lv_result_char = 'u'.
        WHEN 'q'. lv_result_char = 'v'.
        WHEN 'r'. lv_result_char = 'w'.
        WHEN 's'. lv_result_char = 'x'.
        WHEN 't'. lv_result_char = 'y'.
        WHEN 'u'. lv_result_char = 'z'.
        WHEN 'v'. lv_result_char = 'a'.
        WHEN 'w'. lv_result_char = 'b'.
        WHEN 'x'. lv_result_char = 'c'.
        WHEN 'y'. lv_result_char = 'd'.
        WHEN 'z'. lv_result_char = 'e'.
      ENDCASE.

      result = |{ result }{ lv_result_char }|.
    ENDDO.
  ENDMETHOD.

  METHOD decode_shift.
    DATA: lv_char        TYPE c LENGTH 1,
          lv_result_char TYPE c LENGTH 1,
          lv_index       TYPE i.

    DO strlen( s ) TIMES.
      lv_index = sy-index - 1.
      lv_char = s+lv_index(1).

      CASE lv_char.
        WHEN 'f'. lv_result_char = 'a'.
        WHEN 'g'. lv_result_char = 'b'.
        WHEN 'h'. lv_result_char = 'c'.
        WHEN 'i'. lv_result_char = 'd'.
        WHEN 'j'. lv_result_char = 'e'.
        WHEN 'k'. lv_result_char = 'f'.
        WHEN 'l'. lv_result_char = 'g'.
        WHEN 'm'. lv_result_char = 'h'.
        WHEN 'n'. lv_result_char = 'i'.
        WHEN 'o'. lv_result_char = 'j'.
        WHEN 'p'. lv_result_char = 'k'.
        WHEN 'q'. lv_result_char = 'l'.
        WHEN 'r'. lv_result_char = 'm'.
        WHEN 's'. lv_result_char = 'n'.
        WHEN 't'. lv_result_char = 'o'.
        WHEN 'u'. lv_result_char = 'p'.
        WHEN 'v'. lv_result_char = 'q'.
        WHEN 'w'. lv_result_char = 'r'.
        WHEN 'x'. lv_result_char = 's'.
        WHEN 'y'. lv_result_char = 't'.
        WHEN 'z'. lv_result_char = 'u'.
        WHEN 'a'. lv_result_char = 'v'.
        WHEN 'b'. lv_result_char = 'w'.
        WHEN 'c'. lv_result_char = 'x'.
        WHEN 'd'. lv_result_char = 'y'.
        WHEN 'e'. lv_result_char = 'z'.
      ENDCASE.

      result = |{ result }{ lv_result_char }|.
    ENDDO.
  ENDMETHOD.
ENDCLASS.