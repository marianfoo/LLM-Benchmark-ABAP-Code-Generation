CLASS z_humaneval_001 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: separate_paren_groups
      IMPORTING paren_string  TYPE string
      RETURNING VALUE(result) TYPE string_table.

ENDCLASS.

CLASS z_humaneval_001 IMPLEMENTATION.

  METHOD separate_paren_groups.
    DATA: current_string TYPE string,
          current_depth  TYPE i VALUE 0,
          lv_char        TYPE c,
          lv_index       TYPE i.

    DO strlen( paren_string ) TIMES.
      lv_index = sy-index - 1.
      lv_char = paren_string+lv_index(1).

      IF lv_char = '('.
        current_depth = current_depth + 1.
        current_string = current_string && lv_char.
      ELSEIF lv_char = ')'.
        current_depth = current_depth - 1.
        current_string = current_string && lv_char.
        IF current_depth = 0.
          result = VALUE #( BASE result ( current_string ) ).
          CLEAR current_string.
        ENDIF.
      ENDIF.
    ENDDO.
  ENDMETHOD.

ENDCLASS.