CLASS ltcl_test_split_words DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    TYPES: BEGIN OF ty_result,
             strings TYPE string_table,
             count   TYPE i,
           END OF ty_result.

    METHODS: test_hello_world FOR TESTING,
      test_hello_comma_world FOR TESTING,
      test_hello_world_comma FOR TESTING,
      test_multiple_commas FOR TESTING,
      test_abcdef FOR TESTING,
      test_aaabb FOR TESTING,
      test_aaabb_mixed FOR TESTING,
      test_empty_string FOR TESTING.

ENDCLASS.

CLASS ltcl_test_split_words IMPLEMENTATION.

  METHOD test_hello_world.
    DATA(result) = z_humaneval_125=>split_words( 'Hello world!' ).
    cl_abap_unit_assert=>assert_equals( act = lines( result ) exp = 2 ).
    cl_abap_unit_assert=>assert_equals( act = result[ 1 ] exp = 'Hello' ).
    cl_abap_unit_assert=>assert_equals( act = result[ 2 ] exp = 'world!' ).
  ENDMETHOD.

  METHOD test_hello_comma_world.
    DATA(result) = z_humaneval_125=>split_words( 'Hello,world!' ).
    cl_abap_unit_assert=>assert_equals( act = lines( result ) exp = 2 ).
    cl_abap_unit_assert=>assert_equals( act = result[ 1 ] exp = 'Hello' ).
    cl_abap_unit_assert=>assert_equals( act = result[ 2 ] exp = 'world!' ).
  ENDMETHOD.

  METHOD test_hello_world_comma.
    DATA(result) = z_humaneval_125=>split_words( 'Hello world,!' ).
    cl_abap_unit_assert=>assert_equals( act = lines( result ) exp = 2 ).
    cl_abap_unit_assert=>assert_equals( act = result[ 1 ] exp = 'Hello' ).
    cl_abap_unit_assert=>assert_equals( act = result[ 2 ] exp = 'world,!' ).
  ENDMETHOD.

  METHOD test_multiple_commas.
    DATA(result) = z_humaneval_125=>split_words( 'Hello,Hello,world !' ).
    cl_abap_unit_assert=>assert_equals( act = lines( result ) exp = 2 ).
    cl_abap_unit_assert=>assert_equals( act = result[ 1 ] exp = 'Hello,Hello,world' ).
    cl_abap_unit_assert=>assert_equals( act = result[ 2 ] exp = '!' ).
  ENDMETHOD.

  METHOD test_abcdef.
    DATA expected TYPE string_table.
    expected = VALUE #( ( `3` ) ).
    DATA(result) = z_humaneval_125=>split_words( 'abcdef' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_aaabb.
    DATA expected TYPE string_table.
    expected = VALUE #( ( `2` ) ).
    DATA(result) = z_humaneval_125=>split_words( 'aaabb' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_aaabb_mixed.
    DATA expected TYPE string_table.
    expected = VALUE #( ( `1` ) ).
    DATA(result) = z_humaneval_125=>split_words( 'aaaBb' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_empty_string.

    DATA expected TYPE string_table.
    expected = VALUE #(  ).
    DATA(result) = z_humaneval_125=>split_words( '' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

ENDCLASS.