CLASS z_humaneval_054 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: same_chars
      IMPORTING
        s0            TYPE string
        s1            TYPE string
      RETURNING
        VALUE(result) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_054 IMPLEMENTATION.

  METHOD same_chars.
    DATA: lt_chars_s0 TYPE SORTED TABLE OF c WITH UNIQUE KEY table_line,
          lt_chars_s1 TYPE SORTED TABLE OF c WITH UNIQUE KEY table_line,
          lv_len      TYPE i,
          lv_index    TYPE i,
          lv_char     TYPE c.

    lv_len = strlen( s0 ).
    DO lv_len TIMES.
      lv_index = sy-index - 1.
      lv_char = s0+lv_index(1).
      INSERT lv_char INTO TABLE lt_chars_s0.
    ENDDO.

    lv_len = strlen( s1 ).
    DO lv_len TIMES.
      lv_index = sy-index - 1.
      lv_char = s1+lv_index(1).
      INSERT lv_char INTO TABLE lt_chars_s1.
    ENDDO.

    IF lt_chars_s0 = lt_chars_s1.
      result = abap_true.
    ELSE.
      result = abap_false.
    ENDIF.

  ENDMETHOD.

ENDCLASS.