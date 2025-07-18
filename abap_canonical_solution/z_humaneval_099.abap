CLASS z_humaneval_099 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: closest_integer
      IMPORTING
        VALUE(value)  TYPE string
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_099 IMPLEMENTATION.

  METHOD closest_integer.
    DATA: num         TYPE f,
          temp_value  TYPE string,
          decimal_pos TYPE i,
          len         TYPE i,
          offset      TYPE i.

    temp_value = value.

    FIND '.' IN temp_value MATCH OFFSET decimal_pos.
    IF sy-subrc = 0.
      len = strlen( temp_value ).
      WHILE len > 0.
        offset = len - 1.
        IF temp_value+offset(1) = '0'.
          len = len - 1.
          temp_value = temp_value(len).
        ELSE.
          EXIT.
        ENDIF.
      ENDWHILE.
    ENDIF.

    num = temp_value.

    len = strlen( temp_value ).
    IF len >= 2.
      offset = len - 2.
      IF temp_value+offset(2) = '.5'.
        IF num > 0.
          result = trunc( num + 1 ).
        ELSE.
          result = trunc( num - 1 ).
        ENDIF.
      ELSE.
        IF num >= 0.
          result = trunc( num + '0.5' ).
        ELSE.
          result = trunc( num - '0.5' ).
        ENDIF.
      ENDIF.
    ELSEIF len > 0.
      IF num >= 0.
        result = trunc( num + '0.5' ).
      ELSE.
        result = trunc( num - '0.5' ).
      ENDIF.
    ELSE.
      result = 0.
    ENDIF.

  ENDMETHOD.

ENDCLASS.