CLASS z_humaneval_130 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: tri
      IMPORTING
        n             TYPE i
      RETURNING
        VALUE(result) TYPE int4_table.

ENDCLASS.

CLASS z_humaneval_130 IMPLEMENTATION.

  METHOD tri.
    DATA: my_tri     TYPE int4_table,
          i          TYPE i,
          temp_value TYPE i.

    IF n = 0.
      result = VALUE #( ( 1 ) ).
      RETURN.
    ENDIF.

    my_tri = VALUE #( ( 1 ) ( 3 ) ).

    DO n - 1 TIMES.
      i = sy-index + 1.
      IF i MOD 2 = 0.
        temp_value = i / 2 + 1.
        my_tri = VALUE #( BASE my_tri ( temp_value ) ).
      ELSE.
        temp_value = my_tri[ i ] + my_tri[ i - 1 ] + ( i + 3 ) / 2.
        my_tri = VALUE #( BASE my_tri ( temp_value ) ).
      ENDIF.
    ENDDO.

    result = my_tri.
  ENDMETHOD.

ENDCLASS.