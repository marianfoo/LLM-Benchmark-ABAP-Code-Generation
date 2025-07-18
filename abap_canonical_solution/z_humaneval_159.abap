CLASS z_humaneval_159 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: eat
      IMPORTING
        number        TYPE i
        need          TYPE i
        remaining     TYPE i
      RETURNING
        VALUE(result) TYPE int4_table.

ENDCLASS.

CLASS z_humaneval_159 IMPLEMENTATION.

  METHOD eat.
    IF need <= remaining.
      result = VALUE #( ( number + need ) ( remaining - need ) ).
    ELSE.
      result = VALUE #( ( number + remaining ) ( 0 ) ).
    ENDIF.
  ENDMETHOD.

ENDCLASS.