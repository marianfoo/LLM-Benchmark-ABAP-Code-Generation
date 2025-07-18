CLASS z_humaneval_156 DEFINITION PUBLIC.
  PUBLIC SECTION.
    CLASS-METHODS: int_to_mini_roman
      IMPORTING
        number        TYPE i
      RETURNING
        VALUE(result) TYPE string.
ENDCLASS.

CLASS z_humaneval_156 IMPLEMENTATION.
  METHOD int_to_mini_roman.
    DATA: num         TYPE int4_table,
          sym         TYPE string_table,
          i           TYPE i,
          div         TYPE i,
          res         TYPE string,
          temp_number TYPE i,
          current_num TYPE i,
          current_sym TYPE string.

    num = VALUE #( ( 1 ) ( 4 ) ( 5 ) ( 9 ) ( 10 ) ( 40 ) ( 50 ) ( 90 ) ( 100 ) ( 400 ) ( 500 ) ( 900 ) ( 1000 ) ).
    sym = VALUE #( ( `I` ) ( `IV` ) ( `V` ) ( `IX` ) ( `X` ) ( `XL` ) ( `L` ) ( `XC` ) ( `C` ) ( `CD` ) ( `D` ) ( `CM` ) ( `M` ) ).

    i = 13.
    temp_number = number.

    WHILE temp_number > 0 AND i > 0.
      READ TABLE num INDEX i INTO current_num.
      READ TABLE sym INDEX i INTO current_sym.

      IF current_num <= temp_number.
        div = temp_number DIV current_num.
        temp_number = temp_number MOD current_num.

        WHILE div > 0.
          res = res && current_sym.
          div = div - 1.
        ENDWHILE.
      ENDIF.

      i = i - 1.
    ENDWHILE.

    TRANSLATE res TO LOWER CASE.
    result = res.
  ENDMETHOD.
ENDCLASS.