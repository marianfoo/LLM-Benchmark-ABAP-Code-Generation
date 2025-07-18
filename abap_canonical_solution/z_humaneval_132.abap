CLASS z_humaneval_132 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: is_nested
      IMPORTING
        string        TYPE string
      RETURNING
        VALUE(result) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_132 IMPLEMENTATION.

  METHOD is_nested.
    DATA: opening_bracket_index TYPE int4_table,
          closing_bracket_index TYPE int4_table,
          cnt                   TYPE i VALUE 0,
          i                     TYPE i VALUE 0,
          l                     TYPE i,
          idx                   TYPE i.

    DO strlen( string ) TIMES.
      DATA(char_index) = sy-index - 1.
      IF string+char_index(1) = '['.
        opening_bracket_index = VALUE #( BASE opening_bracket_index ( char_index ) ).
      ELSE.
        closing_bracket_index = VALUE #( BASE closing_bracket_index ( char_index ) ).
      ENDIF.
    ENDDO.

    DATA: temp_table TYPE int4_table.
    LOOP AT closing_bracket_index INTO DATA(close_idx) FROM lines( closing_bracket_index ) TO 1 STEP -1.
      temp_table = VALUE #( BASE temp_table ( close_idx ) ).
    ENDLOOP.
    closing_bracket_index = temp_table.

    l = lines( closing_bracket_index ).

    LOOP AT opening_bracket_index INTO idx.
      IF i < l.
        READ TABLE closing_bracket_index INDEX ( i + 1 ) INTO DATA(closing_idx).
        IF idx < closing_idx.
          cnt = cnt + 1.
          i = i + 1.
        ENDIF.
      ENDIF.
    ENDLOOP.

    IF cnt >= 2.
      result = abap_true.
    ELSE.
      result = abap_false.
    ENDIF.

  ENDMETHOD.

ENDCLASS.