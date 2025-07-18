CLASS z_humaneval_089 DEFINITION PUBLIC.
  PUBLIC SECTION.
    CLASS-METHODS encrypt
      IMPORTING
        iv_string           TYPE string
      RETURNING
        VALUE(rv_encrypted) TYPE string.
ENDCLASS.

CLASS z_humaneval_089 IMPLEMENTATION.
  METHOD encrypt.
    DATA: lv_alphabet  TYPE string VALUE 'abcdefghijklmnopqrstuvwxyz',
          lv_out       TYPE string VALUE '',
          lv_char      TYPE c LENGTH 1,
          lv_pos       TYPE i,
          lv_offset    TYPE i,
          lv_new_index TYPE i,
          lv_sub_char  TYPE c LENGTH 1.

    DO strlen( iv_string ) TIMES.
      lv_pos = sy-index - 1.
      lv_char = iv_string+lv_pos(1).

      FIND FIRST OCCURRENCE OF lv_char IN lv_alphabet MATCH OFFSET lv_offset.
      IF sy-subrc = 0.
        lv_new_index = ( lv_offset + 4 ) MOD 26.
        lv_sub_char = lv_alphabet+lv_new_index(1).
        lv_out = lv_out && lv_sub_char.
      ELSE.
        lv_out = lv_out && lv_char.
      ENDIF.
    ENDDO.

    rv_encrypted = lv_out.
  ENDMETHOD.
ENDCLASS.
