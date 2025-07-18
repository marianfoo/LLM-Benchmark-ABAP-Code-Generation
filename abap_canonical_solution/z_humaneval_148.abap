CLASS z_humaneval_148 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: bf
      IMPORTING
        VALUE(planet1) TYPE string
        VALUE(planet2) TYPE string
      RETURNING
        VALUE(result)  TYPE string_table.
ENDCLASS.

CLASS z_humaneval_148 IMPLEMENTATION.
  METHOD bf.
    DATA: planet_names  TYPE string_table,
          planet1_index TYPE i,
          planet2_index TYPE i,
          start_index   TYPE i,
          end_index     TYPE i,
          i             TYPE i.

    planet_names = VALUE #( ( `Mercury` ) ( `Venus` ) ( `Earth` ) ( `Mars` )
                           ( `Jupiter` ) ( `Saturn` ) ( `Uranus` ) ( `Neptune` ) ).

    READ TABLE planet_names TRANSPORTING NO FIELDS WITH KEY table_line = planet1.
    IF sy-subrc <> 0.
      RETURN.
    ENDIF.
    planet1_index = sy-tabix.

    READ TABLE planet_names TRANSPORTING NO FIELDS WITH KEY table_line = planet2.
    IF sy-subrc <> 0.
      RETURN.
    ENDIF.
    planet2_index = sy-tabix.

    IF planet1_index = planet2_index.
      RETURN.
    ENDIF.

    IF planet1_index < planet2_index.
      start_index = planet1_index + 1.
      end_index = planet2_index - 1.
    ELSE.
      start_index = planet2_index + 1.
      end_index = planet1_index - 1.
    ENDIF.

    i = start_index.
    WHILE i <= end_index.
      READ TABLE planet_names INTO DATA(planet) INDEX i.
      result = VALUE #( BASE result ( planet ) ).
      i = i + 1.
    ENDWHILE.
  ENDMETHOD.
ENDCLASS.