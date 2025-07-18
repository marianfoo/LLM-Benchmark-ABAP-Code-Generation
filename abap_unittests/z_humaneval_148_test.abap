CLASS ltcl_test_z_humaneval_148 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_jupiter_neptune FOR TESTING,
      test_earth_mercury FOR TESTING,
      test_mercury_uranus FOR TESTING,
      test_neptune_venus FOR TESTING,
      test_earth_earth FOR TESTING,
      test_mars_earth FOR TESTING,
      test_jupiter_makemake FOR TESTING.
ENDCLASS.

CLASS ltcl_test_z_humaneval_148 IMPLEMENTATION.
  METHOD test_jupiter_neptune.
    DATA: result   TYPE string_table,
          expected TYPE string_table.
    result = z_humaneval_148=>bf( planet1 = `Jupiter` planet2 = `Neptune` ).
    expected = VALUE #( ( `Saturn` ) ( `Uranus` ) ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_earth_mercury.
    DATA: result   TYPE string_table,
          expected TYPE string_table.
    result = z_humaneval_148=>bf( planet1 = `Earth` planet2 = `Mercury` ).
    expected = VALUE #( ( `Venus` ) ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_mercury_uranus.
    DATA: result   TYPE string_table,
          expected TYPE string_table.
    result = z_humaneval_148=>bf( planet1 = `Mercury` planet2 = `Uranus` ).
    expected = VALUE #( ( `Venus` ) ( `Earth` ) ( `Mars` ) ( `Jupiter` ) ( `Saturn` ) ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_neptune_venus.
    DATA: result   TYPE string_table,
          expected TYPE string_table.
    result = z_humaneval_148=>bf( planet1 = `Neptune` planet2 = `Venus` ).
    expected = VALUE #( ( `Earth` ) ( `Mars` ) ( `Jupiter` ) ( `Saturn` ) ( `Uranus` ) ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_earth_earth.
    DATA: result   TYPE string_table,
          expected TYPE string_table.
    result = z_humaneval_148=>bf( planet1 = `Earth` planet2 = `Earth` ).
    expected = VALUE #( ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_mars_earth.
    DATA: result   TYPE string_table,
          expected TYPE string_table.
    result = z_humaneval_148=>bf( planet1 = `Mars` planet2 = `Earth` ).
    expected = VALUE #( ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_jupiter_makemake.
    DATA: result   TYPE string_table,
          expected TYPE string_table.
    result = z_humaneval_148=>bf( planet1 = `Jupiter` planet2 = `Makemake` ).
    expected = VALUE #( ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
    ).
  ENDMETHOD.
ENDCLASS.